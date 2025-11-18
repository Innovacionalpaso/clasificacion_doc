pip install streamlit
pip install PyPDF2
pip install PyMuPDF
pip install python-magic

pip install python-magic-bin


import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from pathlib import Path
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
import pythoncom
import win32com.client as win32
from PIL import Image
import fitz  # PyMuPDF
import magic
import tempfile

# Configuración de la página
st.set_page_config(
    page_title="Clasificador Inteligente de Documentos",
    page_icon="📁",
    layout="wide"
)

class DocumentClassifier:
    def __init__(self):
        self.model = None
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.features = []
        self.labels = []
        
    def extract_features(self, file_path):
        """Extrae características de los archivos"""
        features = {}
        file_stats = os.stat(file_path)
        
        # Características básicas
        features['file_size'] = file_stats.st_size
        features['file_size_kb'] = file_stats.st_size / 1024
        features['file_size_mb'] = file_stats.st_size / (1024 * 1024)
        features['file_extension'] = Path(file_path).suffix.lower()
        features['file_name_length'] = len(Path(file_path).name)
        features['file_name'] = Path(file_path).name
        features['full_path'] = file_path
        features['parent_folder'] = Path(file_path).parent.name
        
        # Tipo MIME
        try:
            mime_type = magic.from_file(file_path, mime=True)
            features['mime_type'] = mime_type
        except:
            features['mime_type'] = 'unknown'
        
        # Para PDFs - características adicionales
        if features['file_extension'] == '.pdf':
            pdf_features = self._analyze_pdf(file_path)
            features.update(pdf_features)
        
        # Clasificar tipo de documento
        features['document_type'] = self._classify_file_type(features['file_extension'])
        
        return features
    
    def _analyze_pdf(self, file_path):
        """Análisis específico para archivos PDF"""
        pdf_features = {
            'has_digital_signature': False,
            'is_image_only': False,
            'has_text': False,
            'page_count': 0,
            'is_scanned': False,
            'summary': '',
            'contains_images': False,
            'contains_forms': False,
            'text_length': 0,
            'image_count': 0
        }
        
        try:
            with fitz.open(file_path) as doc:
                pdf_features['page_count'] = len(doc)
                
                text_content = ""
                image_count = 0
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Extraer texto
                    text = page.get_text()
                    if text.strip():
                        pdf_features['has_text'] = True
                        text_content += text + " "
                    
                    # Contar imágenes
                    image_list = page.get_images()
                    image_count += len(image_list)
                
                pdf_features['image_count'] = image_count
                pdf_features['contains_images'] = image_count > 0
                pdf_features['text_length'] = len(text_content)
                
                # Determinar si es solo imágenes
                if pdf_features['has_text'] and image_count > 0:
                    pdf_features['is_image_only'] = False
                elif not pdf_features['has_text'] and image_count > 0:
                    pdf_features['is_image_only'] = True
                    pdf_features['is_scanned'] = True
                elif not pdf_features['has_text'] and image_count == 0:
                    pdf_features['is_image_only'] = True
                
                # Generar resumen
                if text_content:
                    words = text_content.split()
                    pdf_features['text_length'] = len(words)
                    if len(words) > 100:
                        pdf_features['summary'] = ' '.join(words[:100]) + '...'
                    else:
                        pdf_features['summary'] = text_content
                else:
                    pdf_features['summary'] = 'Documento sin texto extraíble'
                
                # Verificar firma digital (simplificado)
                try:
                    if '/AcroForm' in str(doc._getXrefString(1)):
                        pdf_features['contains_forms'] = True
                except:
                    pass
                
        except Exception as e:
            st.warning(f"Error analizando PDF {file_path}: {str(e)}")
        
        return pdf_features
    
    def scan_directory(self, root_path):
        """Escanea directorio y subdirectorios"""
        documents = []
        total_files = 0
        
        # Crear progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Obtener lista total de archivos para progreso
        all_files = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                all_files.append((root, file))
        
        total_files_count = len(all_files)
        
        for i, (root, file) in enumerate(all_files):
            file_path = os.path.join(root, file)
            
            # Actualizar progreso
            progress = (i + 1) / total_files_count
            progress_bar.progress(progress)
            status_text.text(f"Procesando {i+1}/{total_files_count}: {file}")
            
            try:
                features = self.extract_features(file_path)
                
                document_info = {
                    'file_name': file,
                    'file_path': file_path,
                    'file_size': features.get('file_size', 0),
                    'file_size_kb': features.get('file_size_kb', 0),
                    'file_size_mb': features.get('file_size_mb', 0),
                    'file_extension': features.get('file_extension', ''),
                    'mime_type': features.get('mime_type', ''),
                    'directory': root,
                    'parent_folder': features.get('parent_folder', ''),
                    'last_modified': datetime.fromtimestamp(os.path.getmtime(file_path)),
                    'file_hash': self._calculate_file_hash(file_path),
                    'document_type': features.get('document_type', 'Unknown')
                }
                
                # Información específica para PDFs
                if features['file_extension'] == '.pdf':
                    document_info.update({
                        'has_digital_signature': features.get('has_digital_signature', False),
                        'is_image_only': features.get('is_image_only', False),
                        'has_text': features.get('has_text', False),
                        'page_count': features.get('page_count', 0),
                        'is_scanned': features.get('is_scanned', False),
                        'summary': features.get('summary', ''),
                        'contains_images': features.get('contains_images', False),
                        'contains_forms': features.get('contains_forms', False),
                        'text_length': features.get('text_length', 0),
                        'image_count': features.get('image_count', 0)
                    })
                
                documents.append(document_info)
                
            except Exception as e:
                st.error(f"Error procesando {file_path}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return pd.DataFrame(documents)
    
    def _classify_file_type(self, extension):
        """Clasifica el tipo de archivo por extensión"""
        file_types = {
            '.doc': 'Word', '.docx': 'Word', '.rtf': 'Word',
            '.xls': 'Excel', '.xlsx': 'Excel', '.csv': 'Excel',
            '.ppt': 'PowerPoint', '.pptx': 'PowerPoint',
            '.pdf': 'PDF',
            '.txt': 'Text', '.json': 'Text', '.xml': 'Text',
            '.jpg': 'Image', '.jpeg': 'Image', '.png': 'Image', 
            '.gif': 'Image', '.bmp': 'Image',
            '.zip': 'Archive', '.rar': 'Archive', '.7z': 'Archive',
            '.mp4': 'Video', '.avi': 'Video', '.mkv': 'Video',
            '.mp3': 'Audio', '.wav': 'Audio'
        }
        return file_types.get(extension, 'Other')
    
    def _calculate_file_hash(self, file_path):
        """Calcula hash MD5 del archivo"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return "error"
    
    def prepare_training_data(self, df):
        """Prepara datos para entrenamiento del modelo"""
        # Características para el modelo
        feature_columns = [
            'file_size', 'file_name_length', 'page_count'
        ]
        
        # Convertir variables categóricas
        extensions_encoded = pd.get_dummies(df['file_extension'], prefix='ext')
        mime_types_encoded = pd.get_dummies(df['mime_type'], prefix='mime')
        
        # Combinar características
        X = df[feature_columns].copy()
        X = pd.concat([X, extensions_encoded, mime_types_encoded], axis=1)
        
        # Rellenar NaN
        X = X.fillna(0)
        
        return X
    
    def train_model(self, df):
        """Entrena el modelo de clasificación"""
        st.info("Entrenando modelo de clasificación...")
        
        # Preparar datos
        X = self.prepare_training_data(df)
        y = df['document_type']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Entrenar modelo
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = self.model.predict(X_test)
        
        # Mostrar resultados
        st.subheader("Resultados del Entrenamiento")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Reporte de Clasificación:**")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
        
        with col2:
            st.write("**Matriz de Confusión:**")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=np.unique(y), yticklabels=np.unique(y))
            st.pyplot(fig)
        
        return self.model.score(X_test, y_test)
    
    def predict_document_type(self, file_path):
        """Predice el tipo de documento para un nuevo archivo"""
        if self.model is None:
            st.error("El modelo no ha sido entrenado. Por favor, entrena el modelo primero.")
            return None
        
        features = self.extract_features(file_path)
        features_df = pd.DataFrame([features])
        
        # Preparar características para predicción
        X = self.prepare_training_data(features_df)
        
        # Asegurar que tenga las mismas columnas que el entrenamiento
        missing_cols = set(self.model.feature_names_in_) - set(X.columns)
        for col in missing_cols:
            X[col] = 0
        X = X[self.model.feature_names_in_]
        
        prediction = self.model.predict(X)[0]
        probability = np.max(self.model.predict_proba(X))
        
        return prediction, probability

    def export_to_excel(self, df, filename="matriz_documentos.xlsx"):
        """Exporta la matriz de documentos a Excel con múltiples hojas"""
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Hoja principal con todos los datos
            df.to_excel(writer, sheet_name='Todos los Documentos', index=False)
            
            # Hoja de resumen por tipo
            summary_type = df.groupby('document_type').agg({
                'file_name': 'count',
                'file_size_mb': 'sum',
                'file_size_kb': 'mean'
            }).rename(columns={
                'file_name': 'Cantidad',
                'file_size_mb': 'Tamaño Total (MB)',
                'file_size_kb': 'Tamaño Promedio (KB)'
            }).round(2)
            summary_type.to_excel(writer, sheet_name='Resumen por Tipo')
            
            # Hoja de PDFs detallada
            if 'PDF' in df['document_type'].values:
                pdfs_df = df[df['document_type'] == 'PDF']
                pdfs_df.to_excel(writer, sheet_name='PDFs Detallado', index=False)
                
                # Resumen de PDFs
                pdf_summary = pdfs_df.agg({
                    'page_count': ['count', 'mean', 'sum'],
                    'is_scanned': 'sum',
                    'has_text': 'sum',
                    'contains_images': 'sum'
                }).round(2)
                pdf_summary.to_excel(writer, sheet_name='Resumen PDFs')
            
            # Hoja de estadísticas por carpeta
            folder_stats = df.groupby('parent_folder').agg({
                'file_name': 'count',
                'file_size_mb': 'sum',
                'document_type': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Mixed'
            }).rename(columns={
                'file_name': 'Cantidad Archivos',
                'file_size_mb': 'Tamaño Total (MB)',
                'document_type': 'Tipo Principal'
            }).round(2)
            folder_stats.to_excel(writer, sheet_name='Estadísticas por Carpeta')
        
        return filename

def main():
    st.title("📁 Clasificador Inteligente de Documentos")
    st.markdown("""
    Esta aplicación utiliza machine learning para clasificar y analizar documentos 
    automáticamente. Escanea directorios, extrae características y clasifica los 
    archivos por tipo y contenido.
    """)
    
    # Inicializar clasificador
    if 'classifier' not in st.session_state:
        st.session_state.classifier = DocumentClassifier()
        st.session_state.scan_complete = False
        st.session_state.df_documents = None
    
    classifier = st.session_state.classifier
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    app_mode = st.sidebar.selectbox(
        "Selecciona el modo",
        ["Escaneo de Documentos", "Entrenamiento del Modelo", "Clasificación Nueva", "Análisis y Reportes"]
    )
    
    if app_mode == "Escaneo de Documentos":
        st.header("🔍 Escaneo de Documentos")
        
        # Opciones de selección de carpeta
        st.subheader("Selección de Carpeta")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Método 1: Input de texto para ruta
            st.write("**Método 1: Ingresar Ruta Manualmente**")
            root_path = st.text_input(
                "Ruta del directorio raíz:",
                value=".",
                help="Ruta absoluta o relativa del directorio a escanear"
            )
            
            # Método 2: Selector de carpeta (usando tempfile para demostración)
            st.write("**Método 2: Seleccionar Carpeta**")
            uploaded_folder = st.file_uploader(
                "O sube archivos para analizar una carpeta (selecciona múltiples):",
                type=None,
                accept_multiple_files=True,
                help="Selecciona múltiples archivos para analizar"
            )
        
        with col2:
            st.write("")
            st.write("")
            st.write("")
            scan_button = st.button("🚀 Iniciar Escaneo Completo", type="primary", use_container_width=True)
        
        # Procesar según el método seleccionado
        if scan_button:
            if uploaded_folder and len(uploaded_folder) > 0:
                # Método con archivos subidos
                with st.spinner("Procesando archivos subidos..."):
                    temp_dir = tempfile.mkdtemp()
                    documents = []
                    
                    for uploaded_file in uploaded_folder:
                        # Guardar archivo temporal
                        temp_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        # Extraer características
                        features = classifier.extract_features(temp_path)
                        features['file_name'] = uploaded_file.name
                        features['file_path'] = temp_path
                        features['directory'] = temp_dir
                        features['last_modified'] = datetime.now()
                        features['file_hash'] = classifier._calculate_file_hash(temp_path)
                        
                        documents.append(features)
                    
                    df_documents = pd.DataFrame(documents)
                    st.session_state.df_documents = df_documents
                    st.session_state.scan_complete = True
                    
                    # Limpiar archivos temporales
                    for file in os.listdir(temp_dir):
                        os.remove(os.path.join(temp_dir, file))
                    os.rmdir(temp_dir)
                    
            elif root_path and os.path.exists(root_path):
                # Método con ruta de carpeta
                with st.spinner("Escaneando directorio... Esto puede tomar varios minutos."):
                    df_documents = classifier.scan_directory(root_path)
                    st.session_state.df_documents = df_documents
                    st.session_state.scan_complete = True
            else:
                st.error("Por favor, selecciona una carpeta válida o sube archivos para analizar.")
        
        if st.session_state.scan_complete and st.session_state.df_documents is not None:
            df = st.session_state.df_documents
            
            # Mostrar estadísticas
            st.subheader("📊 Estadísticas del Escaneo")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documentos", len(df))
            
            with col2:
                st.metric("Tipos de Archivo", df['document_type'].nunique())
            
            with col3:
                pdf_count = len(df[df['document_type'] == 'PDF'])
                st.metric("Total PDFs", pdf_count)
            
            with col4:
                st.metric("Tamaño Total", f"{df['file_size_mb'].sum():.2f} MB")
            
            # Distribución de tipos de archivo
            st.subheader("Distribución de Tipos de Documentos")
            fig, ax = plt.subplots(figsize=(10, 6))
            df['document_type'].value_counts().plot(kind='bar', ax=ax)
            plt.xticks(rotation=45)
            plt.title('Cantidad de Documentos por Tipo')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Tabla de documentos
            st.subheader("📋 Matriz de Documentos Encontrados")
            
            # Filtros para la tabla
            col1, col2 = st.columns(2)
            with col1:
                filter_type = st.multiselect(
                    "Filtrar por tipo:",
                    options=df['document_type'].unique(),
                    default=df['document_type'].unique()
                )
            with col2:
                min_size_mb = st.slider("Tamaño mínimo (MB):", 0.0, 100.0, 0.0)
            
            filtered_df = df[
                (df['document_type'].isin(filter_type)) & 
                (df['file_size_mb'] >= min_size_mb)
            ]
            
            st.dataframe(
                filtered_df[[
                    'file_name', 'document_type', 'file_size_mb', 
                    'directory', 'last_modified'
                ]].sort_values('file_size_mb', ascending=False),
                use_container_width=True
            )
            
            # Exportar a Excel
            st.subheader("💾 Exportar Resultados")
            
            if st.button("📊 Exportar Matriz a Excel", type="secondary"):
                with st.spinner("Generando archivo Excel..."):
                    filename = classifier.export_to_excel(df)
                    
                    # Leer el archivo para descarga
                    with open(filename, "rb") as f:
                        excel_data = f.read()
                    
                    st.download_button(
                        label="⬇️ Descargar Archivo Excel",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    # Mostrar preview del Excel
                    st.success("Archivo Excel generado exitosamente!")
                    st.info("El archivo contiene múltiples hojas:")
                    st.write("- **Todos los Documentos**: Matriz completa con todos los datos")
                    st.write("- **Resumen por Tipo**: Estadísticas agrupadas por tipo de documento")
                    st.write("- **PDFs Detallado**: Información específica de archivos PDF")
                    st.write("- **Resumen PDFs**: Estadísticas de documentos PDF")
                    st.write("- **Estadísticas por Carpeta**: Análisis por carpetas")
    
    elif app_mode == "Entrenamiento del Modelo":
        st.header("🤖 Entrenamiento del Modelo")
        
        if not st.session_state.scan_complete:
            st.warning("Por favor, realiza primero el escaneo de documentos.")
        else:
            df = st.session_state.df_documents
            
            st.write(f"Documentos disponibles para entrenamiento: {len(df)}")
            st.write("Distribución de tipos de documentos:")
            st.write(df['document_type'].value_counts())
            
            if st.button("🎯 Entrenar Modelo de Clasificación", type="primary"):
                accuracy = classifier.train_model(df)
                st.session_state.model_trained = True
                st.success(f"Modelo entrenado con exactitud: {accuracy:.2f}")
    
    elif app_mode == "Clasificación Nueva":
        st.header("🎯 Clasificar Nuevos Documentos")
        
        if not hasattr(classifier, 'model') or classifier.model is None:
            st.warning("Por favor, entrena el modelo primero en la pestaña de Entrenamiento.")
        else:
            uploaded_file = st.file_uploader(
                "Sube un documento para clasificar",
                type=['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx', 'txt', 'jpg', 'png', 'csv']
            )
            
            if uploaded_file is not None:
                # Guardar archivo temporal
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Clasificar
                prediction, probability = classifier.predict_document_type(temp_path)
                
                # Mostrar resultados
                st.subheader("Resultados de la Clasificación")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tipo Predicho", prediction)
                with col2:
                    st.metric("Confianza", f"{probability:.2%}")
                
                # Análisis detallado para PDFs
                if uploaded_file.name.lower().endswith('.pdf'):
                    features = classifier._analyze_pdf(temp_path)
                    
                    st.subheader("Análisis Detallado del PDF")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Propiedades:**")
                        st.write(f"- Páginas: {features.get('page_count', 0)}")
                        st.write(f"- Contiene texto: {'Sí' if features.get('has_text') else 'No'}")
                        st.write(f"- Contiene imágenes: {'Sí' if features.get('contains_images') else 'No'}")
                        st.write(f"- Solo imágenes: {'Sí' if features.get('is_image_only') else 'No'}")
                        st.write(f"- Documento escaneado: {'Sí' if features.get('is_scanned') else 'No'}")
                        st.write(f"- Número de imágenes: {features.get('image_count', 0)}")
                    
                    with col2:
                        st.write("**Contenido:**")
                        if features.get('summary'):
                            st.write("**Resumen:**")
                            st.info(features['summary'])
                
                # Limpiar archivo temporal
                os.remove(temp_path)
    
    elif app_mode == "Análisis y Reportes":
        st.header("📈 Análisis y Reportes")
        
        if not st.session_state.scan_complete:
            st.warning("Por favor, realiza primero el escaneo de documentos.")
        else:
            df = st.session_state.df_documents
            
            # Exportación final
            st.subheader("Exportación Completa")
            if st.button("📁 Generar Reporte Completo en Excel", type="primary"):
                with st.spinner("Generando reporte completo..."):
                    filename = classifier.export_to_excel(df, "reporte_completo_documentos.xlsx")
                    
                    with open(filename, "rb") as f:
                        excel_data = f.read()
                    
                    st.download_button(
                        label="📥 Descargar Reporte Completo",
                        data=excel_data,
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

if __name__ == "__main__":
    main()