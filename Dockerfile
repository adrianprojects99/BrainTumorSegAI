FROM python:3.9

WORKDIR /app

# Copiar archivos
COPY requirements.txt .
COPY app7.py .
COPY files/ ./files/
COPY public/ ./public/

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto
EXPOSE 7860

# Comando para ejecutar
CMD ["python", "app7.py"]