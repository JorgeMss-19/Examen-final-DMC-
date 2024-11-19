import streamlit as st
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import torch

# Cargar el extractor de características y el modelo de Hugging Face
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Título de la aplicación
st.title("Clasificador de Imágenes")

# Instrucciones para el usuario
st.write("Sube una imagen para que sea clasificada.")

# Cuadro de subida de imagen
uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Leer la imagen
        image = Image.open(uploaded_file)
        
        # Mostrar la imagen subida
        st.image(image, caption='Imagen subida', use_container_width=True)
        st.write("Clasificando la imagen...")

        # Preprocesar la imagen y prepararla para el modelo
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Realizar la predicción
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Obtener la clase con la mayor probabilidad
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        # Mostrar el resultado
        st.write(f"### Resultado de la Clasificación: {predicted_class}")
        
    except Exception as e:
        st.error(f"Error al procesar la imagen: {e}")