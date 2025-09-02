import cv2
import jetson.inference
import jetson.utils
import numpy as np

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

# Configurar la cámara
camera = cv2.VideoCapture("/dev/video0")
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)

# Umbral de confianza para filtrado
confidence_threshold = 0.5

# Supresión de no máximo (NMS)
def non_max_suppression(detections, overlapThresh=0.3):
    if len(detections) == 0:
        return []

    boxes = np.array([[d.Left, d.Top, d.Right, d.Bottom] for d in detections])
    scores = np.array([d.Confidence for d in detections])
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)
    
    pick = []

    while len(idxs) > 0:
        i = idxs[-1]
        pick.append(i)
        
        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / (areas[idxs[:-1]] + areas[i] - (w * h))
        
        idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > overlapThresh)[0])))
    
    return [detections[i] for i in pick]

while True:
    # Capturar el siguiente frame
    r, frame = camera.read()
    img = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float))
    
    # Detectar objetos en el frame
    detections = net.Detect(img)
    
    # Filtrar detecciones por umbral de confianza
    detections = [d for d in detections if d.Confidence > confidence_threshold]
    
    # Aplicar supresión de no máximo
    detections = non_max_suppression(detections)

    # Crear una imagen de salida para dibujar las detecciones
    output_frame = frame.copy()

    # Dibujar bounding boxes
    for detect in detections:
        Id = detect.ClassID
        item = net.GetClassDesc(Id)
        
        # Seleccionar color basado en la etiqueta
        color = (0, 255, 0) if item == "Ripe" else (0, 0, 255)
        top_left = (int(detect.Left), int(detect.Top))
        bottom_right = (int(detect.Right), int(detect.Bottom))
        
        # Crear un rectángulo semi-transparente
        overlay = output_frame.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, thickness=cv2.FILLED)
        
        # Ajustar la transparencia (misma transparencia para el contorno)
        alpha = 0.3  # Controla la transparencia
        cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
        
        # Añadir el contorno semi-transparente (igual que el relleno)
        cv2.rectangle(overlay, top_left, bottom_right, color, 2)
        cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
        
        # Obtener el porcentaje de confianza y convertirlo a porcentaje
        confidence_text = f"{int(detect.Confidence * 100)}%"
        
        # Añadir el texto de la etiqueta y porcentaje de confianza
        cv2.putText(output_frame, f"{item} ({confidence_text})", 
                    (int(detect.Left), int(detect.Top) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Mostrar el frame con detecciones
    cv2.imshow("window", output_frame)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord("q"):
        break

# Liberar recursos
camera.release()
cv2.destroyAllWindows()

