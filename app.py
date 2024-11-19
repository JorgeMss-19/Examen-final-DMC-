import streamlit as st
from huggingface_hub import InferenceClient
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env 
load_dotenv(dotenv_path=r'config\.env')

# Configuración del token de la API de Hugging Face
api_token = os.getenv("HUGGINGFACE_API_TOKEN")  # token

# Cargar el modelo y procesador de imágenes para clasificación
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Configuración de la app
st.title("Aplicación de Generación y Clasificación de Imágenes con Conexion Funcional")

# Configuración de dos columnas
col1, col2 = st.columns(2)

# Variable para almacenar la imagen generada
generated_image = None

# Columna 1: Generación de imágenes a partir de texto
with col1:
    st.header("Generación de Imágenes a partir de Texto")
    prompt = st.text_input("Ingresa una descripción")

    if st.button("Generar Imagen") and prompt:
        try:
            client = InferenceClient(model="black-forest-labs/FLUX.1-dev", token=api_token)
            response = client.text_to_image(prompt=prompt)

            if response:
                image_path = os.path.join(os.getcwd(), "generated_image.png")
                response.save(image_path)
                generated_image = Image.open(image_path)
                st.image(generated_image, caption="Imagen Generada", use_container_width=True)
            else:
                st.error("No se pudo generar la imagen. Inténtalo de nuevo.")
        except Exception as e:
            st.error(f"Error al generar la imagen: {e}")

# Columna 2: Clasificación de imágenes cargadas o generadas
with col2:
    st.header("Clasificación de Imágenes")

    # Clasificar la imagen generada
    if generated_image is not None:
        st.image(generated_image, caption="Imagen Generada para Clasificación", use_container_width=True)
        st.write("Clasificando la imagen generada...")

        try:
            # Preprocesar la imagen y hacer la predicción
            inputs = image_processor(images=generated_image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Obtener la clase con la mayor probabilidad
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = model.config.id2label[predicted_class_idx]
            st.write(f"### Resultado de la Clasificación: {predicted_class}")
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")

    # Clasificar una imagen cargada
    uploaded_file = st.file_uploader("Elige una imagen para clasificar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Leer y convertir la imagen al modo RGB
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Imagen subida", use_container_width=True)
            st.write("Clasificando la imagen subida...")

            # Preprocesar la imagen y hacer la predicción
            inputs = image_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits

            # Obtener la clase con la mayor probabilidad
            predicted_class_idx = logits.argmax(-1).item()
            predicted_class = model.config.id2label[predicted_class_idx]
            st.write(f"### Resultado de la Clasificación: {predicted_class}")
        except Exception as e:
            st.error(f"Error al procesar la imagen: {e}")
