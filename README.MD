# 🧠 Brain Tumor Segmentation AI

Sistema de segmentación automática de tumores cerebrales usando arquitectura U-Net.

## 🎓 Universidad Latina de Panamá
**Autores:** Adrian Herrera, Brian Chiru  
**Profesor:** Ing. Rangel Isaias Alvarado Walles  
**Curso:** Inteligencia Artificial  

## 📊 Resultados
- **Dice Coefficient:** 87%
- **IoU:** 79%
- **Precisión:** 94%

## 🛠️ Instalación Local

\`\`\`bash
# Clonar repositorio
git clone https://github.com/TU-USUARIO/BrainTumorSegAI.git
cd BrainTumorSegAI

# Instalar Git LFS (si no lo tienes)
git lfs install
git lfs pull

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
python app7.py
\`\`\`

## 📁 Estructura del Proyecto

\`\`\`
BrainTumorSegAI/
├── app.py              # Aplicación web PyWebIO
├── requirements.txt    # Dependencias
├── Dockerfile         # Para deployment
├── files/
│   └── unet.h5       # Modelo entrenado (Git LFS)
└── public/
    └── ULatina.png   # Logo institucional
\`\`\`


## Despliegue del Proyecto

Debido a las limitaciones de almacenamiento para archivos grandes (el modelo `unet.h5`), la aplicación completa y el modelo entrenado se han subido a **Hugging Face Spaces**.

Puede acceder a la demostración funcional en el siguiente enlace:

👉 **[Ver Proyecto en Hugging Face](https://huggingface.co/spaces/adexherr/bts)**

## 🔬 Stack Tecnológico

- **Framework:** TensorFlow/Keras
- **Interfaz:** PyWebIO
- **Procesamiento:** OpenCV, NumPy
- **Deployment:** Hugging Face Spaces (Docker)

## 📄 Licencia
MIT License

## 📧 Contacto
- Adrian Herrera: aherrera@est.ulatina.edu.pa

- Brian Chiru: bchiru@est.ulatina.edu.pa
