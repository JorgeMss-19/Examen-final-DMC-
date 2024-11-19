from huggingface_hub import InferenceClient
import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env 
load_dotenv(dotenv_path='config\.env')

def generate_image(prompt):
    api_token = os.getenv("HUGGINGFACE_API_TOKEN")  
    client = InferenceClient(model="black-forest-labs/FLUX.1-dev", token=api_token)

    response = client.text_to_image(prompt=prompt)

    # Verificar la respuesta
    if response:
        # Guardar la imagen en una ruta específica
        image_path = os.path.join(os.getcwd(), "test_generated_image.png")
        response.save(image_path)
        return image_path
    else:
        return None

# Probar la función
if __name__ == "__main__":
    input_text = "Un paisaje de montaña al atardecer"
    image_path = generate_image(input_text)
    print("Imagen generada y guardada en:", image_path)