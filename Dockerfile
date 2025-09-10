# Usa una imagen base de Python 3.8
FROM python:3.8

# Define el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo de dependencias
COPY ./requirements.txt ./requirements.txt

# Instala las dependencias
RUN pip install --no-cache-dir --upgrade -r ./requirements.txt

# Copia el resto de los archivos necesarios
COPY ./app.py .
COPY ./model ./model
COPY ./data ./data

# Expone el puerto 80
EXPOSE 80

# Comando para levantar el servidor con uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
