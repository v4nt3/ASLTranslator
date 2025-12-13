"""
Interfaz web con Streamlit para detección de lenguaje de señas
Soporta webcam en vivo y upload de videos
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
from pathlib import Path
import time
from pipelines.inference.realtime_sign_detector import SignLanguageDetector

st.set_page_config(
    page_title="Detector de Lenguaje de Señas",
    page_icon="🤟",
    layout="wide"
)

# Inicializar detector en session state
@st.cache_resource
def load_detector(model_path, classes_path, feature_dim):
    return SignLanguageDetector(
        model_path=model_path,
        class_names_path=classes_path,
        feature_dim=feature_dim,
        device="cuda",
        confidence_threshold=0.3
    )


def main():
    st.title(" Detector de Lenguaje de Señas")
    st.markdown("Detecta señas continuas formando oraciones en tiempo real")
    
    # Sidebar - Configuración
    st.sidebar.header("Configuración")
    
    model_path = st.sidebar.text_input(
        "Ruta al modelo (.pth)",
        value="checkpoints/temporal82/best_model.pth"
    )
    
    classes_path = st.sidebar.text_input(
        "Ruta a clases (.json)",
        value="data/dataset_metadata.json"
    )
    
    feature_dim = st.sidebar.selectbox(
        "Dimensión de features",
        [1152, 640],
        index=0,
        help="1152 para ResNet1024+MLP128, 640 para ResNet512+MLP128"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Umbral de confianza",
        0.0, 1.0, 0.3, 0.06
    )
    
    # Modo de operación
    mode = st.sidebar.radio(
        "Modo de detección",
        [" Webcam en vivo", " Subir video"]
    )
    
    # Cargar detector
    try:
        detector = load_detector(model_path, classes_path, feature_dim)
        detector.confidence_threshold = confidence_threshold
        st.sidebar.success(" Modelo cargado")
    except Exception as e:
        st.sidebar.error(f"Error cargando modelo: {e}")
        return
    
    # Layout principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Video")
        video_placeholder = st.empty()
    
    with col2:
        st.header("Detecciones")
        sentence_placeholder = st.empty()
        detections_placeholder = st.empty()
    
    # Modo Webcam
    if mode == " Webcam en vivo":
        st.info("Presiona 'Iniciar Webcam' para comenzar. Presiona 'q' en la ventana de video para detener.")
        
        if st.button(" Iniciar Webcam"):
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("No se pudo acceder a la webcam")
                return
            
            # Reiniciar buffers
            detector.frame_buffer.clear()
            detector.keypoint_buffer.clear()
            
            sentence = []
            detections_list = []
            last_sign = None
            last_sign_time = 0
            frame_count = 0
            start_time = time.time()
            
            stop_button = st.button("⏹Detener")
            
            while not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                current_time = time.time() - start_time
                
                # Procesar frame
                success = detector.process_frame(frame)
                
                # Predicción cada 12 frames
                if frame_count % 12 == 0 and success:
                    prediction = detector.predict_window()
                    
                    if prediction:
                        sign_name, confidence = prediction
                        
                        if sign_name != last_sign or (current_time - last_sign_time) > 2.0:
                            detections_list.append((sign_name, confidence, current_time))
                            sentence.append(sign_name)
                            last_sign = sign_name
                            last_sign_time = current_time
                
                # Dibujar anotaciones
                if last_sign and (current_time - last_sign_time) < 3.0:
                    cv2.putText(
                        frame,
                        f"{last_sign} ({confidence:.2f})",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
                
                # Mostrar frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                
                # Actualizar detecciones
                sentence_text = " ".join(sentence[-15:])
                sentence_placeholder.markdown(f"### Oración: **{sentence_text}**")
                
                if detections_list:
                    detection_text = "\n".join([
                        f"- **{sign}** ({conf:.2f}) @ {ts:.1f}s"
                        for sign, conf, ts in detections_list[-10:]
                    ])
                    detections_placeholder.markdown(detection_text)
            
            cap.release()
            st.success(f" Detección completada. {len(sentence)} señas detectadas.")
    
    # Modo Upload
    else:
        uploaded_file = st.file_uploader(
            "Sube un video",
            type=["mp4", "avi", "mov", "mkv"]
        )
        
        if uploaded_file is not None:
            # Guardar video temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            st.info("Procesando video...")
            
            # Procesar video
            detections_list, sentence = detector.process_video_stream(
                video_source=video_path,
                display=False
            )
            
            # Mostrar resultados
            st.success(f" Procesamiento completado")
            
            sentence_text = " ".join(sentence)
            st.markdown(f"### Oración detectada:")
            st.markdown(f"## **{sentence_text}**")
            
            st.markdown(f"###  Detecciones ({len(detections_list)}):")
            
            if detections_list:
                detection_df = [
                    {
                        "Seña": sign,
                        "Confianza": f"{conf:.2%}",
                        "Tiempo (s)": f"{ts:.2f}"
                    }
                    for sign, conf, ts in detections_list
                ]
                st.table(detection_df)
            
            # Limpiar archivo temporal
            Path(video_path).unlink()


if __name__ == "__main__":
    main()
