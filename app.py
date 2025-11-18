# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import io

# Configuración de la página
st.set_page_config(layout="wide", page_title="Predicción de Salud Financiera", page_icon="📊")

# --- ESTILOS Y CSS PERSONALIZADO ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
        text-align: center;
    }
    .health-excellent { 
        background: linear-gradient(135deg, #d4edda 0%, #a8e6b8 100%);
        border: 3px solid #28a745;
    }
    .health-good { 
        background: linear-gradient(135deg, #d1ecf1 0%, #a8dce8 100%);
        border: 3px solid #17a2b8;
    }
    .health-regular { 
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
        border: 3px solid #ffc107;
    }
    .health-deficient { 
        background: linear-gradient(135deg, #f8d7da 0%, #f5c2c7 100%);
        border: 3px solid #dc3545;
    }
    .health-critical { 
        background: linear-gradient(135deg, #dc3545 0%, #a02a37 100%);
        border: 3px solid #721c24;
        color: white;
    }
    .indicator-box {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
    }
    .score-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 0.3rem;
    }
    .badge-green { background-color: #28a745; color: white; }
    .badge-yellow { background-color: #ffc107; color: #333; }
    .badge-red { background-color: #dc3545; color: white; }
    .prediction-card {
        background-color: #f8f9fa;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .chart-container {
        background-color: transparent;
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- FUNCIÓN PARA CALCULAR SALUD FINANCIERA ---
def calcular_salud_financiera(morosidad, liquidez, roa, roe, cobertura):
    """
    Calcula el índice de salud financiera basado en los rangos definidos
    Retorna: puntaje_total, categoria, descripcion, clase_css, icono, desglose_puntajes
    """
    def puntuar_indicador(valor, rangos_verde, rangos_amarillo, tipo):
        if tipo == "menor_mejor":  # Para morosidad (menor es mejor)
            if valor < rangos_verde:
                return 100
            elif valor <= rangos_amarillo:
                return 50
            else:
                return 0
        else:  # Para los demás (mayor es mejor)
            if valor > rangos_verde:
                return 100
            elif valor >= rangos_amarillo:
                return 50
            else:
                return 0
    
    # Pesos de cada indicador
    pesos = {
        'morosidad': 0.30,
        'liquidez': 0.25,
        'roa': 0.20,
        'roe': 0.15,
        'cobertura': 0.10
    }
    
    # Calcular puntajes individuales
    p_morosidad = puntuar_indicador(morosidad, 5, 10, "menor_mejor")
    p_liquidez = puntuar_indicador(liquidez, 20, 10, "mayor_mejor")
    p_roa = puntuar_indicador(roa, 1.5, 0.5, "mayor_mejor")
    p_roe = puntuar_indicador(roe, 15, 10, "mayor_mejor")
    p_cobertura = puntuar_indicador(cobertura, 120, 100, "mayor_mejor")
    
    # Calcular puntajes ponderados
    pond_morosidad = p_morosidad * pesos['morosidad']
    pond_liquidez = p_liquidez * pesos['liquidez']
    pond_roa = p_roa * pesos['roa']
    pond_roe = p_roe * pesos['roe']
    pond_cobertura = p_cobertura * pesos['cobertura']
    
    # Puntaje total
    puntaje_total = pond_morosidad + pond_liquidez + pond_roa + pond_roe + pond_cobertura
    
    # Determinar categoría de salud (3 zonas)
    if puntaje_total >= 70:
        categoria = "ZONA VERDE"
        descripcion = "Salud financiera óptima"
        clase_css = "health-excellent"
        icono = "🟢"
    elif puntaje_total >= 40:
        categoria = "ZONA AMARILLA"
        descripcion = "Salud financiera aceptable con áreas de mejora"
        clase_css = "health-regular"
        icono = "🟡"
    else:
        categoria = "ZONA ROJA"
        descripcion = "Situación crítica, requiere intervención inmediata"
        clase_css = "health-critical"
        icono = "🔴"
    
    # Desglose detallado
    desglose = {
        'Morosidad': {'base': p_morosidad, 'ponderado': pond_morosidad, 'peso': pesos['morosidad']},
        'Liquidez': {'base': p_liquidez, 'ponderado': pond_liquidez, 'peso': pesos['liquidez']},
        'ROA': {'base': p_roa, 'ponderado': pond_roa, 'peso': pesos['roa']},
        'ROE': {'base': p_roe, 'ponderado': pond_roe, 'peso': pesos['roe']},
        'Cobertura': {'base': p_cobertura, 'ponderado': pond_cobertura, 'peso': pesos['cobertura']}
    }
    
    return puntaje_total, categoria, descripcion, clase_css, icono, desglose

def obtener_clase_badge(puntaje_base):
    """Retorna la clase CSS para el badge según el puntaje base"""
    if puntaje_base == 100:
        return "badge-green"
    elif puntaje_base == 50:
        return "badge-yellow"
    else:
        return "badge-red"

# --- FUNCIÓN PARA CREAR GRÁFICOS ESTILIZADOS SIN FONDO ---
def crear_grafico_limpio(figsize=(12, 6)):
    """Configura matplotlib para gráficos sin fondo"""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.tick_params(colors='#333333')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    return fig, ax

# --- TÍTULO PRINCIPAL ---
st.markdown('<h1 class="main-header">🔮 Predicción de Salud Financiera - Cooperativas</h1>', unsafe_allow_html=True)

# --- CARGA DE MODELO Y ESCALADOR ---
@st.cache_resource
def load_prediction_assets():
    try:
        model = load_model("modelo_general_lstm.h5")
        scaler = joblib.load("scaler_general.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error al cargar recursos: {e}")
        return None, None

model, scaler = load_prediction_assets()

if model is None or scaler is None:
    st.stop()

# --- DEFINICIÓN DE INDICADORES ---
indicadores_elegidos_raw = [
    'MOROSIDAD DE LA CARTERA TOTAL',
    'COBERTURA DE LA CARTERA PROBLEMÁTICA',
    'RESULTADOS DEL EJERCICIO / ACTIVO PROMEDIO',
    'FONDOS DISPONIBLES / TOTAL DEPOSITOS A CORTO PLAZO ',
    'RESULTADOS DEL EJERCICIO / PATRIMONIO PROMEDIO'
]

indicadores_renamed = {
    'MOROSIDAD DE LA CARTERA TOTAL': 'MOROSIDAD',
    'COBERTURA DE LA CARTERA PROBLEMÁTICA': 'COBERTURA_CARTERA_PROBLEMATICA',
    'RESULTADOS DEL EJERCICIO / ACTIVO PROMEDIO': 'ROA',
    'FONDOS DISPONIBLES / TOTAL DEPOSITOS A CORTO PLAZO ': 'LIQUIDEZ',
    'RESULTADOS DEL EJERCICIO / PATRIMONIO PROMEDIO': 'ROE'
}

indicadores_model_order = ['COBERTURA_CARTERA_PROBLEMATICA', 'LIQUIDEZ', 'MOROSIDAD', 'ROA', 'ROE']

# --- INTERFAZ DE CARGA DE ARCHIVOS ---
st.markdown('<h2 class="sub-header">📁 Carga de Datos Históricos</h2>', unsafe_allow_html=True)

col_upload, col_info = st.columns([2, 1])

with col_upload:
    archivo = st.file_uploader("Sube tu archivo Excel con datos históricos", type=["xlsx"])

with col_info:
    st.info("💡 El archivo debe contener al menos 12 meses de datos históricos para realizar predicciones precisas.")

df_processed = None
if archivo is not None:
    with st.spinner("🔄 Procesando archivo..."):
        try:
            df_raw = pd.read_excel(io.BytesIO(archivo.getvalue()))
            df_raw['RUC'] = df_raw['RUC'].astype(str)
            
            # Procesamiento de datos
            df_filtered = df_raw[df_raw['INDICADOR_FINANCIERO'].isin(indicadores_elegidos_raw)].copy()
            
            for ind in indicadores_elegidos_raw:
                mean_val = df_filtered[df_filtered['INDICADOR_FINANCIERO'] == ind]['VALOR'].mean()
                df_filtered.loc[df_filtered['INDICADOR_FINANCIERO'] == ind, 'VALOR'] = \
                    df_filtered.loc[df_filtered['INDICADOR_FINANCIERO'] == ind, 'VALOR'].fillna(mean_val)
            
            df_processed = df_filtered.pivot_table(
                index=['RUC', 'FECHA_CORTE'], 
                columns='INDICADOR_FINANCIERO', 
                values='VALOR'
            ).reset_index()
            df_processed.columns.name = None
            df_processed = df_processed.rename(columns=indicadores_renamed)
            
            if all(col in df_processed.columns for col in indicadores_model_order):
                df_processed['FECHA_CORTE'] = pd.to_datetime(df_processed['FECHA_CORTE'])
                df_processed = df_processed[['RUC', 'FECHA_CORTE'] + indicadores_model_order].sort_values(['RUC', 'FECHA_CORTE'])
                
                # Convertir decimales a porcentajes
                for col in indicadores_model_order:
                    df_processed[col] = df_processed[col] * 100
                
                # Calcular salud financiera histórica
                historico_salud = []
                for _, row in df_processed.iterrows():
                    puntaje, categoria, descripcion, clase_css, icono, desglose = calcular_salud_financiera(
                        row['MOROSIDAD'], row['LIQUIDEZ'], row['ROA'], row['ROE'], row['COBERTURA_CARTERA_PROBLEMATICA']
                    )
                    historico_salud.append(puntaje)
                
                df_processed['SALUD_FINANCIERA'] = historico_salud
                
                st.success("✅ Datos procesados correctamente")
                
                # Mostrar preview de datos
                with st.expander("👁️ Vista previa de datos procesados"):
                    st.dataframe(df_processed.head(10), use_container_width=True)
            else:
                st.error("❌ Faltan columnas necesarias en los datos")
                df_processed = None
                
        except Exception as e:
            st.error(f"❌ Error procesando archivo: {e}")
            df_processed = None

# --- INTERFAZ DE PREDICCIÓN ---
if df_processed is not None:
    st.markdown('<h2 class="sub-header">🎯 Configuración de Predicción</h2>', unsafe_allow_html=True)
    
    rucs_disponibles = df_processed['RUC'].unique()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ruc_input = st.selectbox("🏢 Selecciona RUC de la cooperativa", rucs_disponibles)
    with col2:
        future_steps = st.slider("📅 Meses a predecir", 1, 12, 6)
    with col3:
        st.write("")
        st.write("")
        predict_button = st.button("🚀 Generar Predicción", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("🔮 Realizando predicción y análisis..."):
            df_ruc = df_processed[df_processed['RUC'] == ruc_input].sort_values('FECHA_CORTE')
            
            if len(df_ruc) < 12:
                st.warning("⚠️ Se necesitan al menos 12 meses de datos históricos")
            else:
                # Preparar datos para predicción (convertir de % a decimal)
                data = df_ruc[indicadores_model_order].values / 100
                data_scaled = scaler.transform(data)
                time_step = 12
                last_sequence = data_scaled[-time_step:]
                future_preds_original = []
                
                # Generar predicciones
                for _ in range(future_steps):
                    input_seq = last_sequence.reshape(1, time_step, len(indicadores_model_order))
                    pred_scaled = model.predict(input_seq, verbose=0)
                    pred_original = scaler.inverse_transform(pred_scaled)[0]
                    
                    # Asegurar valores positivos para cobertura
                    idx_cobertura = indicadores_model_order.index('COBERTURA_CARTERA_PROBLEMATICA')
                    pred_original[idx_cobertura] = max(0.0, pred_original[idx_cobertura])
                    
                    future_preds_original.append(pred_original)
                    last_sequence = np.append(last_sequence[1:], pred_scaled, axis=0)
                
                # Crear DataFrame con predicciones (convertir a %)
                last_date = df_ruc['FECHA_CORTE'].max()
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                           periods=future_steps, freq='MS')
                
                df_future = pd.DataFrame(future_preds_original, 
                                       index=future_dates, 
                                       columns=indicadores_model_order) * 100
                
                # Calcular salud financiera para predicciones
                salud_futura = []
                for _, row in df_future.iterrows():
                    puntaje, categoria, descripcion, clase, icono, desglose = calcular_salud_financiera(
                        row['MOROSIDAD'], row['LIQUIDEZ'], row['ROA'], row['ROE'], 
                        row['COBERTURA_CARTERA_PROBLEMATICA']
                    )
                    salud_futura.append({
                        'Fecha': row.name,
                        'Puntaje': puntaje,
                        'Categoría': categoria,
                        'Descripción': descripcion,
                        'Clase': clase,
                        'Icono': icono,
                        'Desglose': desglose
                    })
                
                # --- VISUALIZACIÓN DE RESULTADOS ---
                st.markdown("---")
                st.markdown('<h2 class="sub-header">📊 Resultados de la Predicción</h2>', unsafe_allow_html=True)
                
                # === GRÁFICO DE EVOLUCIÓN DE SALUD FINANCIERA ===
                st.markdown("### 📈 Evolución del Índice de Salud Financiera")
                
                fig_salud, ax_salud = crear_grafico_limpio(figsize=(14, 6))
                
                # Datos históricos
                ax_salud.plot(df_ruc['FECHA_CORTE'], df_ruc['SALUD_FINANCIERA'], 
                            label='Histórico', color='#2E86AB', marker='o', linewidth=2.5, 
                            markersize=6, markeredgecolor='white', markeredgewidth=1.5)
                
                # Datos predichos
                fechas_pred = [s['Fecha'] for s in salud_futura]
                puntajes_pred = [s['Puntaje'] for s in salud_futura]
                ax_salud.plot(fechas_pred, puntajes_pred, 
                            label='Predicción', color='#A23B72', marker='s', linestyle='--', 
                            linewidth=2.5, markersize=7, markeredgecolor='white', markeredgewidth=1.5)
                
                # Línea divisoria
                ax_salud.axvline(x=last_date, color='#F18F01', linestyle=':', 
                               linewidth=2.5, label='Inicio Predicción', alpha=0.8)
                
                # Áreas de categorías (3 zonas)
                ax_salud.axhspan(70, 100, alpha=0.2, color='green', label='Zona Verde: Óptima')
                ax_salud.axhspan(40, 70, alpha=0.2, color='yellow', label='Zona Amarilla: Aceptable')
                ax_salud.axhspan(0, 40, alpha=0.2, color='red', label='Zona Roja: Crítica')
                
                ax_salud.set_ylabel('Puntaje de Salud Financiera', fontsize=12, fontweight='bold', color='#333333')
                ax_salud.set_xlabel('Fecha', fontsize=12, fontweight='bold', color='#333333')
                ax_salud.set_title('Evolución del Índice de Salud Financiera', 
                                  fontsize=14, fontweight='bold', color='#1f77b4', pad=20)
                ax_salud.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
                              fancybox=True, shadow=True, fontsize=10)
                ax_salud.set_ylim(0, 100)
                
                plt.tight_layout()
                st.pyplot(fig_salud)
                plt.close()
                
                # === CUADROS DETALLADOS POR MES ===
                st.markdown("---")
                st.markdown("### 📋 Análisis Detallado por Mes Predicho")
                
                for i, (fecha, pred) in enumerate(zip(fechas_pred, future_preds_original)):
                    pred_pct = pred * 100  # Convertir a porcentaje
                    salud_mes = salud_futura[i]
                    desglose = salud_mes['Desglose']
                    
                    with st.container():
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3 style="color: #1f77b4; margin-bottom: 1rem;">📅 {fecha.strftime('%B %Y')}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Tres columnas: Indicadores | Puntajes | Salud General
                        col_ind, col_scores, col_health = st.columns([2, 2, 1.5])
                        
                        with col_ind:
                            st.markdown("**📊 Indicadores Predichos:**")
                            
                            indicadores_display = {
                                'MOROSIDAD': ('🔴 Morosidad', pred_pct[indicadores_model_order.index('MOROSIDAD')]),
                                'LIQUIDEZ': ('💧 Liquidez', pred_pct[indicadores_model_order.index('LIQUIDEZ')]),
                                'ROA': ('📈 ROA', pred_pct[indicadores_model_order.index('ROA')]),
                                'ROE': ('💰 ROE', pred_pct[indicadores_model_order.index('ROE')]),
                                'COBERTURA_CARTERA_PROBLEMATICA': ('🛡️ Cobertura', pred_pct[indicadores_model_order.index('COBERTURA_CARTERA_PROBLEMATICA')])
                            }
                            
                            for key, (nombre, valor) in indicadores_display.items():
                                st.markdown(f"""
                                <div class="indicator-box">
                                    <strong>{nombre}:</strong> <span style="font-size: 1.1rem; color: #2E86AB;">{valor:.2f}%</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col_scores:
                            st.markdown("**🎯 Puntajes por Indicador:**")
                            
                            puntajes_display = {
                                'Morosidad': desglose['Morosidad'],
                                'Liquidez': desglose['Liquidez'],
                                'ROA': desglose['ROA'],
                                'ROE': desglose['ROE'],
                                'Cobertura': desglose['Cobertura']
                            }
                            
                            for nombre, info in puntajes_display.items():
                                badge_class = obtener_clase_badge(info['base'])
                                st.markdown(f"""
                                <div class="indicator-box">
                                    <strong>{nombre}:</strong> 
                                    <span class="score-badge {badge_class}">{info['base']:.0f} pts</span>
                                    <span style="color: #666; font-size: 0.9rem;">
                                        → Ponderado: {info['ponderado']:.2f} pts (Peso: {info['peso']*100:.0f}%)
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col_health:
                            st.markdown("**🏆 Salud Financiera:**")
                            st.markdown(
                                f"""
                                <div class="metric-card {salud_mes['Clase']}" style="margin-top: 0;">
                                    <h1 style="margin: 0; font-size: 3rem;">{salud_mes['Icono']}</h1>
                                    <h2 style="margin: 0.5rem 0; font-size: 2rem;">{salud_mes['Puntaje']:.1f}</h2>
                                    <p style="margin: 0.3rem 0; font-size: 1rem; font-weight: bold;">{salud_mes['Categoría']}</p>
                                    <p style="margin: 0.3rem 0; font-size: 0.85rem;">{salud_mes['Descripción']}</p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                            
                            # Recomendación
                            recomendaciones = {
                                "ZONA VERDE": ("✅", "Mantener estándares actuales"),
                                "ZONA AMARILLA": ("⚠️", "Implementar mejoras específicas"),
                                "ZONA ROJA": ("🚨", "Acción correctiva urgente")
                            }
                            
                            icono_rec, texto_rec = recomendaciones[salud_mes['Categoría']]
                            st.info(f"{icono_rec} **Recomendación:** {texto_rec}")
                        
                        st.markdown("---")
                
                # === GRÁFICOS INDIVIDUALES DE INDICADORES ===
                st.markdown("### 📊 Tendencia Detallada de Indicadores")
                
                fig_indicadores, axes = plt.subplots(2, 3, figsize=(16, 10))
                fig_indicadores.patch.set_alpha(0.0)
                axes = axes.flatten()
                
                colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
                nombres_amigables = {
                    'COBERTURA_CARTERA_PROBLEMATICA': '🛡️ Cobertura de Cartera',
                    'LIQUIDEZ': '💧 Liquidez',
                    'MOROSIDAD': '🔴 Morosidad',
                    'ROA': '📈 ROA',
                    'ROE': '💰 ROE'
                }
                
                for idx, indicador in enumerate(indicadores_model_order):
                    if idx < len(axes):
                        ax = axes[idx]
                        ax.patch.set_alpha(0.0)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        ax.spines['left'].set_color('#CCCCCC')
                        ax.spines['bottom'].set_color('#CCCCCC')
                        ax.tick_params(colors='#333333')
                        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                        
                        # Histórico
                        ax.plot(df_ruc['FECHA_CORTE'], df_ruc[indicador], 
                               label='Histórico', color=colors[idx], linewidth=2.5, 
                               marker='o', markersize=5, markeredgecolor='white', markeredgewidth=1)
                        
                        # Predicción
                        ax.plot(df_future.index, df_future[indicador], 
                               label='Predicción', color=colors[idx], linestyle='--', 
                               linewidth=2.5, marker='s', markersize=6, 
                               markeredgecolor='white', markeredgewidth=1, alpha=0.8)
                        
                        ax.axvline(x=last_date, color='#F18F01', linestyle=':', 
                                 alpha=0.7, linewidth=2)
                        
                        ax.set_title(nombres_amigables.get(indicador, indicador), 
                                   fontweight='bold', fontsize=11, color='#333333', pad=10)
                        ax.legend(loc='best', fontsize=9, frameon=True, fancybox=True, shadow=True)
                        ax.tick_params(axis='x', rotation=45, labelsize=9)
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
                
                # Ocultar último subplot
                if len(indicadores_model_order) < len(axes):
                    axes[-1].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig_indicadores)
                plt.close()

# --- INFORMACIÓN ADICIONAL ---
st.markdown("---")

col_info1, col_info2 = st.columns(2)

with col_info1:
    with st.expander("ℹ️ Metodología del Índice de Salud Financiera"):
        st.markdown("""
        ### 📊 Cálculo del Índice
        
        El índice se calcula ponderando 5 indicadores clave:
        
        | Indicador | Peso | Zona Verde | Zona Amarilla | Zona Roja |
        |-----------|------|------------|---------------|-----------|
        | **Morosidad** | 30% | < 5% | 5% - 10% | > 10% |
        | **Liquidez** | 25% | > 20% | 10% - 20% | < 10% |
        | **ROA** | 20% | > 1.5% | 0.5% - 1.5% | < 0.5% |
        | **ROE** | 15% | > 15% | 10% - 15% | < 10% |
        | **Cobertura** | 10% | > 120% | 100% - 120% | < 100% |
        
        **Fórmula:**
        ```
        Salud = (Morosidad × 30%) + (Liquidez × 25%) + (ROA × 20%) + (ROE × 15%) + (Cobertura × 10%)
        ```
        
        **Clasificación Final:**
        - 🟢 **ZONA VERDE (70-100 pts):** Salud financiera óptima
        - 🟡 **ZONA AMARILLA (40-69 pts):** Salud financiera aceptable con áreas de mejora
        - 🔴 **ZONA ROJA (0-39 pts):** Situación crítica, requiere intervención inmediata
        """)

with col_info2:
    with st.expander("📄 Formato del Archivo Excel Requerido"):
        st.markdown("""
        ### 📋 Estructura de Datos
        
        El archivo Excel debe contener las siguientes columnas:
        
        | Columna | Descripción | Ejemplo |
        |---------|-------------|---------|
        | **RUC** | Identificador de la cooperativa | "1234567890001" |
        | **FECHA_CORTE** | Fecha del período | "2024-01-31" |
        | **INDICADOR_FINANCIERO** | Nombre del indicador | "MOROSIDAD DE LA CARTERA TOTAL" |
        | **VALOR** | Valor del indicador (decimal) | 0.0345 (representa 3.45%) |
        
        **Indicadores requeridos:**
        1. MOROSIDAD DE LA CARTERA TOTAL
        2. COBERTURA DE LA CARTERA PROBLEMÁTICA
        3. RESULTADOS DEL EJERCICIO / ACTIVO PROMEDIO
        4. FONDOS DISPONIBLES / TOTAL DEPOSITOS A CORTO PLAZO
        5. RESULTADOS DEL EJERCICIO / PATRIMONIO PROMEDIO
        
        **Notas importantes:**
        - Los valores deben estar en formato decimal (0.05 = 5%)
        - Se requieren al menos 12 meses de datos históricos
        - Las fechas deben estar en formato de fecha Excel
        """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>💡 <strong>Sistema de Predicción de Salud Financiera para Cooperativas</strong></p>
    <p style="font-size: 0.9rem;">Desarrollado con IA avanzada (LSTM) para análisis predictivo financiero</p>
</div>
""", unsafe_allow_html=True)
