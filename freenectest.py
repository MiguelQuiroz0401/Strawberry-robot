import freenect
import numpy as np
import cv2
import jetson.inference
import jetson.utils

def obtener_video():
    # Obtener datos de video del Kinect
    return freenect.sync_get_video()[0]

# Cargar el modelo de detección
net = jetson.inference.detectNet(
    "ssd-mobilenet-v2",
    [
        "--model=models/fresas100/ssd-mobilenet.onnx",
        "--labels=models/fresas100/labels.txt",
        "--input-blob=input_0",
        "--output-cvg=scores",
        "--output-bbox=boxes"
    ]
)

def main():
    # Inicializar freenect
    freenect.init()
    dispositivo = freenect.open_device(freenect.init(), 1)

    while True:
        try:
            # Obtener imagen de color del Kinect
            imagen_color = obtener_video()

            # Convertir la imagen de BGR (por defecto) a RGB (para Jetson)
            imagen_color_rgb = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2RGB)
            
            # Convertir la imagen de numpy a CUDA para Jetson
            img_cuda = jetson.utils.cudaFromNumpy(imagen_color_rgb)

            # Detectar objetos en el frame
            detections = net.Detect(img_cuda)

            # Crear una imagen de salida para dibujar las detecciones
            output_frame = imagen_color.copy()

            # Dibujar bounding boxes
            for detect in detections:
                if detect.Confidence > 0.5:  # Filtrar detecciones por umbral de confianza
                    top_left = (int(detect.Left), int(detect.Top))
                    bottom_right = (int(detect.Right), int(detect.Bottom))
                    
                    # Seleccionar color basado en la etiqueta
                    color = (0, 255, 0) if net.GetClassDesc(detect.ClassID) == "Ripe" else (0, 0, 255)
                    
                    # Crear un rectángulo semi-transparente
                    overlay = output_frame.copy()
                    cv2.rectangle(overlay, top_left, bottom_right, color, thickness=cv2.FILLED)
                    
                    # Ajustar la transparencia
                    alpha = 0.3
                    cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
                    
                    # Añadir el contorno semi-transparente
                    cv2.rectangle(overlay, top_left, bottom_right, color, 2)
                    cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
                    
                    # Añadir el texto de la etiqueta y porcentaje de confianza
                    confidence_text = f"{int(detect.Confidence * 100)}%"
                    cv2.putText(output_frame, f"{net.GetClassDesc(detect.ClassID)} ({confidence_text})", 
                                (int(detect.Left), int(detect.Top) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Mostrar el frame con detecciones
            cv2.imshow("Imagen de Color con Detección", output_frame)

            # Salir si se presiona la tecla 'q'
            if cv2.waitKey(1) == ord('q'):
                break

        except Exception as e:
            print(f"Error: {e}")
            break

    # Liberar recursos
    cv2.destroyAllWindows()
    freenect.sync_stop()

if __name__ == "__main__":
    main()

