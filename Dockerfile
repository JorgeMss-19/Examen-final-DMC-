# Usa una imagen base de Python
FROM python:3.9-slim

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos de requirements.txt al contenedor
COPY requirements.txt /app/

# Instala las dependencias necesarias
RUN pip install --no-cache-dir -r requirements.txt

# Copia el contenido de la aplicación al directorio de trabajo
COPY . /app/

# Expone el puerto 8000
EXPOSE 8000

# Comando para ejecutar la aplicación en el puerto 8000
CMD ["streamlit", "run", "app.py", "--server.port=8000", "--server.address=0.0.0.0"]
