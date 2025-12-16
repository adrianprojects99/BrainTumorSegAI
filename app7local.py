from pywebio.input import file_upload, actions
from pywebio.output import (put_image, put_text, put_markdown, put_row, 
                             put_loading, put_html, clear, put_button, toast)
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from PIL import Image
import io
import os
from datetime import datetime

# --- 1. DEFINICIÓN DE FUNCIONES ---
def dice_loss(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice = (2. * intersection + 1e-15) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1e-15)
    return 1.0 - dice

# Variable global para el modelo
MODEL_PATH = "files/unet.h5"
LOGO_PATH = "public/ULatina.png"
model = None

def cargar_modelo():
    """Carga el modelo una sola vez"""
    global model
    if model is None:
        try:
            with CustomObjectScope({'dice_loss': dice_loss}):
                model = tf.keras.models.load_model(MODEL_PATH)
            return True, "✅ Modelo cargado correctamente."
        except Exception as e:
            return False, f"❌ Error cargando modelo: {e}"
    return True, "✅ Modelo ya cargado."

def mostrar_encabezado():
    """Muestra el encabezado con logo y título"""
    # Intentar cargar el logo
    logo_existe = os.path.exists(LOGO_PATH)
    
    # CSS personalizado para limpiar estilos y eliminar espacios
    put_html("""
        <style>
            body {
                margin: 0;
                padding: 0;
            }
            .pywebio_output_container {
                padding-top: 0 !important;
            }
            .header-container {
                text-align: center;
                padding: 20px;
                background: white;
                margin: 0;
            }
            .logo-container {
                margin-bottom: 15px;
            }
            .logo-container img {
                max-width: 250px;
                height: auto;
            }
            .title {
                color: #333;
                font-size: 2em;
                font-weight: bold;
                margin: 10px 0;
            }
            .subtitle {
                color: #666;
                font-size: 1.1em;
                margin-bottom: 0;
            }
            .divider {
                border-top: 2px solid #e0e0e0;
                margin: 20px 0;
            }
        </style>
    """)
    
    put_html('<div class="header-container">')
    
    # Mostrar logo si existe
    if logo_existe:
        try:
            with open(LOGO_PATH, 'rb') as f:
                logo_img = Image.open(f)
                put_html('<div class="logo-container">')
                put_image(logo_img)
                put_html('</div>')
        except Exception as e:
            put_html('<div class="logo-container"><p style="color: #999; font-size: 1.2em; font-weight: bold;">Universidad Latina de Panamá</p></div>')
    else:
        put_html('<div class="logo-container"><p style="color: #999; font-size: 1.2em; font-weight: bold;">Universidad Latina de Panamá</p></div>')
    
    put_html("""
        <div class="title"> Detección de Tumores Cerebrales</div>
        <div class="subtitle">Sistema de Segmentación con U-NET | Herramienta de Apoyo al Diagnóstico</div>
    </div>
    <div class="divider"></div>
    """)

def calcular_estadisticas(mascara, img_original):
    """Calcula estadísticas básicas de la segmentación"""
    # Dimensiones totales
    total_pixeles = mascara.shape[0] * mascara.shape[1]
    pixeles_tumor = np.sum(mascara > 127)  # Píxeles blancos (tumor detectado)
    
    # Detectar área del cerebro (excluyendo fondo negro)
    # Convertir imagen a escala de grises
    img_gray = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
    
    # Usar threshold para separar cerebro de fondo negro
    # Píxeles > 10 se consideran parte del cerebro (ajustable según el dataset)
    _, brain_mask = cv2.threshold(img_gray, 10, 255, cv2.THRESH_BINARY)
    
    # Aplicar operaciones morfológicas para limpiar la máscara
    kernel = np.ones((5,5), np.uint8)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
    brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
    
    pixeles_cerebro = np.sum(brain_mask > 127)
    
    # Calcular porcentajes
    porcentaje_total = (pixeles_tumor / total_pixeles) * 100 if total_pixeles > 0 else 0
    porcentaje_cerebro = (pixeles_tumor / pixeles_cerebro) * 100 if pixeles_cerebro > 0 else 0
    
    return {
        'total_pixeles': total_pixeles,
        'pixeles_cerebro': pixeles_cerebro,
        'pixeles_tumor': pixeles_tumor,
        'porcentaje_total': porcentaje_total,
        'porcentaje_cerebro': porcentaje_cerebro,
        'brain_mask': brain_mask
    }

def procesar_imagen(img_file):
    """Procesa la imagen y retorna los resultados"""
    content = img_file['content']
    
    # Convertir bytes a imagen OpenCV
    nparr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, "❌ Error: No se pudo leer la imagen. Intenta con otro archivo."
    
    h_orig, w_orig, _ = img.shape

    # Preprocesar
    img_resized = cv2.resize(img, (256, 256))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    # Predecir
    prediccion = model.predict(img_input, verbose=0)[0]

    # Post-procesar
    mascara = cv2.resize(prediccion, (w_orig, h_orig))
    mascara = (mascara > 0.5).astype(np.uint8) * 255

    # Convertir a RGB antes de calcular estadísticas
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Calcular estadísticas (ahora incluye detección de área cerebral)
    stats = calcular_estadisticas(mascara, img_rgb)

    # Crear Overlay (Mapa de calor)
    heatmap = cv2.applyColorMap(mascara, cv2.COLORMAP_JET)
    resultado = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Convertir imágenes finales
    resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
    mascara_rgb = cv2.cvtColor(mascara, cv2.COLOR_GRAY2RGB)

    # Convertir a formato que PyWebIO entiende (PIL Image)
    img_pil = Image.fromarray(img_rgb)
    res_pil = Image.fromarray(resultado_rgb)
    mask_pil = Image.fromarray(mascara_rgb)
    brain_mask_pil = Image.fromarray(cv2.cvtColor(stats['brain_mask'], cv2.COLOR_GRAY2RGB))

    return {
        'original': img_pil,
        'resultado': res_pil,
        'mascara': mask_pil,
        'brain_mask': brain_mask_pil,
        'stats': stats,
        'dimensiones': (h_orig, w_orig)
    }, None

def mostrar_resultados(resultados):
    """Muestra los resultados del análisis"""
    put_markdown("---")
    put_markdown("###  Resultados del Análisis")
    
    # Información de la imagen
    dims = resultados['dimensiones']
    stats = resultados['stats']
    
    put_html(f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <strong> Dimensiones:</strong> {dims[1]} x {dims[0]} píxeles<br>
            <strong> Área cerebral detectada:</strong> {stats['pixeles_cerebro']:,} píxeles ({(stats['pixeles_cerebro']/stats['total_pixeles']*100):.1f}% de la imagen)<br>
            <strong> Fecha de análisis:</strong> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}
        </div>
    """)
    
    # Mostrar imágenes (ahora incluye máscara del cerebro)
    put_markdown("#### Comparación de Imágenes")
    put_row([
        put_image(resultados['original']).style('width: 100%; border: 3px solid #4CAF50; border-radius: 8px;'),
        put_image(resultados['resultado']).style('width: 100%; border: 3px solid #2196F3; border-radius: 8px;')
    ])
    
    put_html("""
        <div style="display: flex; justify-content: space-around; margin: 15px 0; text-align: center;">
            <div style="flex: 1;"><strong> Imagen Original</strong></div>
            <div style="flex: 1;"><strong> Segmentación IA</strong></div>
        </div>
    """)
    
    put_row([
        put_image(resultados['mascara']).style('width: 100%; border: 3px solid #FF9800; border-radius: 8px;'),
        put_image(resultados['brain_mask']).style('width: 100%; border: 3px solid #9C27B0; border-radius: 8px;')
    ])
    
    put_html("""
        <div style="display: flex; justify-content: space-around; margin: 15px 0 25px 0; text-align: center;">
            <div style="flex: 1;"><strong> Máscara del Tumor</strong></div>
            <div style="flex: 1;"><strong> Área Cerebral Detectada</strong></div>
        </div>
    """)
    
    # Estadísticas mejoradas
    put_markdown("####  Estadísticas de Segmentación")
    
    # Interpretar resultado basado en porcentaje cerebral (más preciso)
    porcentaje = stats['porcentaje_cerebro']
    if porcentaje < 0.5:
        interpretacion = "✅ No se detectaron regiones significativas"
        color = "#4CAF50"
        nivel = "Negativo"
    elif porcentaje < 3:
        interpretacion = "⚠️ Región de interés detectada (área pequeña)"
        color = "#FF9800"
        nivel = "Leve"
    elif porcentaje < 10:
        interpretacion = "⚠️ Región de interés detectada (área moderada)"
        color = "#FF6F00"
        nivel = "Moderado"
    else:
        interpretacion = "🔴 Región de interés detectada (área considerable)"
        color = "#f44336"
        nivel = "Significativo"
    
    put_html(f"""
        <div style="background: {color}; color: white; padding: 20px; border-radius: 8px; margin: 15px 0;">
            <h3 style="margin: 0 0 10px 0;">{interpretacion}</h3>
            <div style="font-size: 1.1em;">
                <strong>Nivel de detección:</strong> {nivel}
            </div>
        </div>
    """)
    
    # Tabla comparativa de métricas
    put_html(f"""
        <div style="background: white; border: 2px solid #e0e0e0; border-radius: 8px; padding: 20px; margin: 15px 0;">
            <h4 style="margin-top: 0;"> Métricas Detalladas</h4>
            <table style="width: 100%; border-collapse: collapse;">
                <tr style="background: #f5f5f5;">
                    <th style="padding: 12px; text-align: left; border-bottom: 2px solid #ddd;">Métrica</th>
                    <th style="padding: 12px; text-align: right; border-bottom: 2px solid #ddd;">Valor</th>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">
                        <strong> Área del tumor (respecto al cerebro)</strong><br>
                        <small style="color: #666;">Métrica más precisa - excluye fondo negro</small>
                    </td>
                    <td style="padding: 10px; text-align: right; border-bottom: 1px solid #eee;">
                        <strong style="font-size: 1.3em; color: {color};">{porcentaje:.2f}%</strong>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">
                        <strong> Área del tumor (respecto a imagen total)</strong><br>
                        <small style="color: #666;">Incluye fondo negro - menos precisa</small>
                    </td>
                    <td style="padding: 10px; text-align: right; border-bottom: 1px solid #eee;">
                        {stats['porcentaje_total']:.2f}%
                    </td>
                </tr>
                <tr>
                    <td style="padding: 10px; border-bottom: 1px solid #eee;">
                        <strong> Píxeles de tejido cerebral</strong>
                    </td>
                    <td style="padding: 10px; text-align: right; border-bottom: 1px solid #eee;">
                        {stats['pixeles_cerebro']:,}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 10px;">
                        <strong> Píxeles de tumor detectados</strong>
                    </td>
                    <td style="padding: 10px; text-align: right;">
                        {stats['pixeles_tumor']:,}
                    </td>
                </tr>
            </table>
        </div>
    """)
    
    # Disclaimer
    put_html("""
        <div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; border-radius: 5px;">
            <strong>⚠️ Importante:</strong> Esta es una herramienta de apoyo al diagnóstico. 
            Los resultados deben ser interpretados por un profesional médico calificado. 
            No reemplaza el juicio clínico ni los métodos de diagnóstico establecidos.
        </div>
    """)

def main():
    while True:
        clear()  # Limpia la pantalla para nueva ejecución
        
        # Mostrar encabezado
        mostrar_encabezado()
        
        # Cargar modelo si no está cargado
        put_text("🔄 Inicializando sistema de IA...")
        exito, mensaje = cargar_modelo()
        
        if exito:
            put_text(mensaje).style('color: green; font-weight: bold;')
        else:
            put_text(mensaje).style('color: red; font-weight: bold;')
            put_markdown("---")
            put_text("❌ Por favor, verifica que el archivo 'files/unet.h5' existe y es válido.")
            put_button("🔄 Reintentar", onclick=lambda: None, color='primary')
            break
        
        put_markdown("---")
        put_markdown("### 📤 Cargar Imagen MRI")

        # Entrada del usuario
        img_file = file_upload("Selecciona una imagen MRI para análisis", accept="image/*")
        
        if img_file:
            # Procesamiento
            with put_loading():
                toast("🔍 Analizando imagen...", duration=2)
                resultados, error = procesar_imagen(img_file)
            
            if error:
                put_text(error).style('color: red; font-weight: bold;')
            else:
                # Mostrar resultados
                mostrar_resultados(resultados)
                toast("✅ Análisis completado", duration=3, color='success')
            
            put_markdown("---")
            
            # Pregunta al usuario si quiere analizar otra imagen
            accion = actions(
                label="¿Qué deseas hacer?",
                buttons=[
                    {'label': '🔄 Analizar otra imagen', 'value': 'continuar'},
                    {'label': '🚪 Salir', 'value': 'salir'}
                ]
            )
            
            if accion == 'salir':
                clear()
                put_html("""
                    <div style="text-align: center; padding: 50px;">
                        <h2>¡Gracias por usar el sistema!</h2>
                        <p>Universidad Latina de Panamá</p>
                        <p style="color: #666;">Puedes cerrar esta ventana</p>
                    </div>
                """)
                break
            # Si continuar, el loop reinicia automáticamente

if __name__ == '__main__':
    from pywebio import start_server
    
    # Configuración del servidor con archivos estáticos
    start_server(
        main, 
        port=8080, 
        debug=True,
        static_dir='public'  # Sirve archivos desde la carpeta public
    )