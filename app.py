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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .health-excellent { background-color: #d4edda; border-left-color: #28a745; }
    .health-good { background-color: #d1ecf1; border-left-color: #17a2b8; }
    .health-regular { background-color: #fff3cd; border-left-color: #ffc107; }
    .health-deficient { background-color: #f8d7da; border-left-color: #dc3545; }
    .health-critical { background-color: #dc3545; border-left-color: #721c24; color: white; }
</style>
""", unsafe_allow_html=True)

# --- FUNCIÓN PARA CALCULAR SALUD FINANCIERA ---
def calcular_salud_financiera(morosidad, liquidez, roa, roe, cobertura):
    """
    Calcula el índice de salud financiera basado en los rangos definidos
    """
    def puntuar_indicador(valor, rangos_verde, rangos_amarillo, tipo):
        if tipo == "menor_mejor":  # Para morosidad (menor es mejor)
            if valor < rangos_verde:
                return 100
            elif valor <= rangos_amarillo:
                return 60
            else:
                return 20
        else:  # Para los demás (mayor es mejor)
            if valor > rangos_verde:
                return 100
            elif valor >= rangos_amarillo:
                return 60
            else:
                return 20
    
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
    
    # Calcular puntaje total ponderado
    puntaje_total = (
        p_morosidad * pesos['morosidad'] +
        p_liquidez * pesos['liquidez'] +
        p_roa * pesos['roa'] +
        p_roe * pesos['roe'] +
        p_cobertura * pesos['cobertura']
    )
    
    # Determinar categoría de salud
    if puntaje_total >= 85:
        categoria = "Excelente"
        clase_css = "health-excellent"
    elif puntaje_total >= 70:
        categoria = "Buena"
        clase_css = "health-good"
    elif puntaje_total >= 55:
        categoria = "Regular"
        clase_css = "health-regular"
    elif puntaje_total >= 40:
        categoria = "Deficiente"
        clase_css = "health-deficient"
    else:
        categoria = "Crítica"
        clase_css = "health-critical"
    
    return puntaje_total, categoria, clase_css

# --- FUNCIÓN PARA CREAR GRÁFICOS ESTILIZADOS ---
def crear_grafico_estilizado():
    """Configura el estilo de matplotlib para los gráficos"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
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
st.header("📁 Carga de Datos Históricos")
archivo = st.file_uploader("Sube tu archivo Excel con datos históricos", type=["xlsx"])

df_processed = None
if archivo is not None:
    with st.spinner("Procesando archivo..."):
        try:
            df_raw = pd.read_excel(io.BytesIO(archivo.getvalue()))
            df_raw['RUC'] = df_raw['RUC'].astype(str)
            
            # Procesamiento de datos (mantener tu lógica existente)
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
                
                # Calcular salud financiera histórica
                historico_salud = []
                for _, row in df_processed.iterrows():
                    puntaje, categoria, _ = calcular_salud_financiera(
                        row['MOROSIDAD'], row['LIQUIDEZ'], row['ROA'], row['ROE'], row['COBERTURA_CARTERA_PROBLEMATICA']
                    )
                    historico_salud.append(puntaje)
                
                df_processed['SALUD_FINANCIERA'] = historico_salud
                
                st.success("✅ Datos procesados correctamente")
                st.dataframe(df_processed.head(), use_container_width=True)
            else:
                st.error("❌ Faltan columnas necesarias en los datos")
                df_processed = None
                
        except Exception as e:
            st.error(f"❌ Error procesando archivo: {e}")
            df_processed = None

# --- INTERFAZ DE PREDICCIÓN ---
if df_processed is not None:
    rucs_disponibles = df_processed['RUC'].unique()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        ruc_input = st.selectbox("Selecciona RUC para predecir", rucs_disponibles)
    with col2:
        future_steps = st.slider("Meses a predecir", 1, 12, 6)
    
    if st.button("🚀 Generar Predicción y Análisis", type="primary"):
        with st.spinner("Realizando predicción..."):
            df_ruc = df_processed[df_processed['RUC'] == ruc_input].sort_values('FECHA_CORTE')
            
            if len(df_ruc) < 12:
                st.warning("⚠️ Se necesitan al menos 12 meses de datos históricos")
            else:
                # Preparar datos para predicción
                data = df_ruc[indicadores_model_order].values
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
                
                # Crear DataFrame con predicciones
                last_date = df_ruc['FECHA_CORTE'].max()
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                           periods=future_steps, freq='M')
                
                df_future = pd.DataFrame(future_preds_original, 
                                       index=future_dates, 
                                       columns=indicadores_model_order)
                
                # Calcular salud financiera para predicciones
                salud_futura = []
                for _, row in df_future.iterrows():
                    puntaje, categoria, clase = calcular_salud_financiera(
                        row['MOROSIDAD'], row['LIQUIDEZ'], row['ROA'], row['ROE'], 
                        row['COBERTURA_CARTERA_PROBLEMATICA']
                    )
                    salud_futura.append({
                        'Fecha': row.name,
                        'Puntaje': puntaje,
                        'Categoría': categoria,
                        'Clase': clase
                    })
                
                # --- VISUALIZACIÓN DE RESULTADOS ---
                st.header("📊 Resultados de la Predicción")
                
                # Gráfico de Evolución de Salud Financiera
                st.subheader("📈 Evolución de la Salud Financiera")
                fig_salud, ax_salud = crear_grafico_estilizado()
                
                # Datos históricos de salud
                ax_salud.plot(df_ruc['FECHA_CORTE'], df_ruc['SALUD_FINANCIERA'], 
                            label='Histórico', color='blue', marker='o', linewidth=2)
                
                # Datos predichos de salud
                fechas_pred = [s['Fecha'] for s in salud_futura]
                puntajes_pred = [s['Puntaje'] for s in salud_futura]
                ax_salud.plot(fechas_pred, puntajes_pred, 
                            label='Predicción', color='red', marker='s', linestyle='--', linewidth=2)
                
                ax_salud.axvline(x=last_date, color='green', linestyle=':', 
                               linewidth=2, label='Inicio Predicción')
                
                # Líneas de referencia para categorías
                ax_salud.axhline(y=85, color='green', linestyle='-', alpha=0.3, label='Excelente (85+)')
                ax_salud.axhline(y=70, color='blue', linestyle='-', alpha=0.3, label='Buena (70-84)')
                ax_salud.axhline(y=55, color='orange', linestyle='-', alpha=0.3, label='Regular (55-69)')
                ax_salud.axhline(y=40, color='red', linestyle='-', alpha=0.3, label='Deficiente (40-54)')
                
                ax_salud.set_ylabel('Puntaje de Salud Financiera')
                ax_salud.set_xlabel('Fecha')
                ax_salud.set_title('Evolución del Índice de Salud Financiera')
                ax_salud.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax_salud.grid(True, alpha=0.3)
                ax_salud.set_ylim(0, 100)
                
                st.pyplot(fig_salud)
                
                # --- CUADRO DE PREDICCIONES POR MES ---
                st.subheader("📋 Detalle de Predicciones por Mes")
                
                for i, (fecha, pred) in enumerate(zip(fechas_pred, future_preds_original)):
                    salud_mes = salud_futura[i]
                    
                    # Crear columnas para cada mes predicho
                    with st.container():
                        st.markdown(f"### 📅 {fecha.strftime('%B %Y')}")
                        
                        col_pred1, col_pred2, col_pred3 = st.columns([2, 1, 1])
                        
                        with col_pred1:
                            # Mostrar valores de indicadores
                            st.markdown("**Indicadores Predichos:**")
                            indicadores_data = {
                                'Indicador': ['Morosidad', 'Liquidez', 'ROA', 'ROE', 'Cobertura'],
                                'Valor': [
                                    f"{pred[indicadores_model_order.index('MOROSIDAD')]:.2f}%",
                                    f"{pred[indicadores_model_order.index('LIQUIDEZ')]:.2f}%",
                                    f"{pred[indicadores_model_order.index('ROA')]:.2f}%",
                                    f"{pred[indicadores_model_order.index('ROE')]:.2f}%",
                                    f"{pred[indicadores_model_order.index('COBERTURA_CARTERA_PROBLEMATICA')]:.2f}%"
                                ]
                            }
                            st.dataframe(pd.DataFrame(indicadores_data), use_container_width=True)
                        
                        with col_pred2:
                            # Mostrar puntaje de salud
                            st.markdown("**Salud Financiera:**")
                            st.markdown(
                                f"""
                                <div class="metric-card {salud_mes['Clase']}">
                                    <h3>{salud_mes['Puntaje']:.1f} pts</h3>
                                    <p><strong>{salud_mes['Categoría']}</strong></p>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                        
                        with col_pred3:
                            # Recomendación basada en la categoría
                            st.markdown("**Recomendación:**")
                            recomendaciones = {
                                "Excelente": "✅ Continuar con la gestión actual",
                                "Buena": "📈 Mantener estrategias, buscar mejoras menores",
                                "Regular": "⚠️ Revisar procesos, identificar áreas de mejora",
                                "Deficiente": "🔴 Plan de acción correctivo necesario",
                                "Crítica": "🚨 Intervención urgente requerida"
                            }
                            st.info(recomendaciones[salud_mes['Categoría']])
                        
                        st.markdown("---")
                
                # --- GRÁFICOS INDIVIDUALES DE INDICADORES ---
                st.subheader("📊 Tendencia de Indicadores Individuales")
                
                # Configurar subplots
                fig_indicadores, axes = plt.subplots(2, 3, figsize=(15, 10))
                axes = axes.flatten()
                
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
                
                for idx, indicador in enumerate(indicadores_model_order):
                    if idx < len(axes):
                        ax = axes[idx]
                        
                        # Nombre amigable para el título
                        nombre_amigable = next(
                            (k for k, v in indicadores_renamed.items() if v == indicador), 
                            indicador
                        )
                        
                        # Datos históricos
                        ax.plot(df_ruc['FECHA_CORTE'], df_ruc[indicador], 
                               label='Histórico', color=colors[idx], linewidth=2, marker='o')
                        
                        # Datos predichos
                        ax.plot(df_future.index, df_future[indicador], 
                               label='Predicción', color=colors[idx], linestyle='--', 
                               linewidth=2, marker='s')
                        
                        ax.axvline(x=last_date, color='green', linestyle=':', alpha=0.7)
                        ax.set_title(nombre_amigable, fontweight='bold')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        ax.tick_params(axis='x', rotation=45)
                        
                        # Formatear eje Y con porcentajes
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}%'))
                
                # Ocultar el último subplot si no se usa
                if len(indicadores_model_order) < len(axes):
                    axes[-1].set_visible(False)
                
                plt.tight_layout()
                st.pyplot(fig_indicadores)

# --- INFORMACIÓN ADICIONAL ---
with st.expander("ℹ️ Información sobre el Índice de Salud Financiera"):
    st.markdown("""
    ### 📊 Métodología del Índice de Salud Financiera
    
    El índice se calcula ponderando 5 indicadores clave:
    
    | Indicador | Peso | Zona Verde | Zona Amarilla | Zona Roja |
    |-----------|------|------------|---------------|-----------|
    | **Morosidad** | 30% | < 5% | 5% - 10% | > 10% |
    | **Liquidez** | 25% | > 20% | 10% - 20% | < 10% |
    | **ROA** | 20% | > 1.5% | 0.5% - 1.5% | < 0.5% |
    | **ROE** | 15% | > 15% | 10% - 15% | < 10% |
    | **Cobertura** | 10% | > 120% | 100% - 120% | < 100% |
    
    **Escala de evaluación:**
    - 🟢 **Excelente (85-100 pts):** Todas las áreas en zona verde
    - 🔵 **Buena (70-84 pts):** Mayoría en zona verde, algunas en amarilla  
    - 🟡 **Regular (55-69 pts):** Mezcla de zonas, necesita atención
    - 🟠 **Deficiente (40-54 pts):** Múltiples áreas en zona roja
    - 🔴 **Crítica (<40 pts):** Situación de alto riesgo
    """)
