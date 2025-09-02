import cv2
import numpy as np
import jetson.inference
import jetson.utils
import time

# Definir umbral de confianza
confidence_threshold = 0.5

# Función para supresión de no máximo (NMS)
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

# Cargar el modelo RGB
net_rgb = jetson.inference.detectNet(
    "ssd-mobilenet-v2",
    [
        "--model=models/fresas100/ssd-mobilenet.onnx",
        "--labels=models/fresas100/labels.txt",
        "--input-blob=input_0",
        "--output-cvg=scores",
        "--output-bbox=boxes"
    ]
)

# Cargar el modelo HSV
net_hsv = jetson.inference.detectNet(
    "ssd-mobilenet-v2",
    [
        "--model=models/hsv/ssd-mobilenet.onnx",
        "--labels=models/hsv/labels.txt",
        "--input-blob=input_0",
        "--output-cvg=scores",
        "--output-bbox=boxes"
    ]
)

# Inicializar la cámara CSI con GStreamer
camera_csi_1 = cv2.VideoCapture(2)#"nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)320, height=(int)240, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

if not camera_csi_1.isOpened():
    print("Error: No se puede acceder a la cámara CSI.")
    exit()

# Obtener las dimensiones de la cámara (resolución de entrada)
frame_width = int(camera_csi_1.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera_csi_1.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Crear ventanas para visualización con un tamaño adecuado (más grandes)
cv2.namedWindow('RGB Model', cv2.WINDOW_NORMAL)
cv2.resizeWindow('RGB Model', 640, 480)  # Ventana más grande para el modelo RGB

cv2.namedWindow('HSV Model', cv2.WINDOW_NORMAL)
cv2.resizeWindow('HSV Model', 640, 480)  # Ventana más grande para el modelo HSV

print('Press \'q\' to stop')

try:
    while True:
        # Capturar frame de la cámara CSI
        ret, frame = camera_csi_1.read()
        if not ret:
            print("Error: No se pudo obtener el frame de la cámara CSI.")
            break

        # Convertir a numpy array para procesamiento con OpenCV
        frame_np = np.array(frame)

        # Convertir la imagen a HSV
        hsv_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2HSV)

        # Usar la imagen RGB para el modelo RGB
        img_rgb = jetson.utils.cudaFromNumpy(frame_np.astype(np.float32))
        detections_rgb = net_rgb.Detect(img_rgb)
        detections_rgb = [d for d in detections_rgb if d.Confidence > confidence_threshold]
        detections_rgb = non_max_suppression(detections_rgb)

        # Usar la imagen HSV para el modelo HSV
        img_hsv = jetson.utils.cudaFromNumpy(hsv_frame.astype(np.float32))
        detections_hsv = net_hsv.Detect(img_hsv)
        detections_hsv = [d for d in detections_hsv if d.Confidence > confidence_threshold]
        detections_hsv = non_max_suppression(detections_hsv)

        # Crear una imagen de salida para cada modelo
        output_rgb = frame_np.copy()  # Imagen RGB original para el modelo RGB
        output_hsv = frame_np.copy()  # Imagen HSV original para el modelo HSV

        # Dibujar bounding boxes para el modelo RGB
        for detect in detections_rgb:
            Id = detect.ClassID
            item = net_rgb.GetClassDesc(Id)
            color = (0, 255, 0) if item == "Ripe" else (0, 0, 255)
            top_left = (int(detect.Left), int(detect.Top))
            bottom_right = (int(detect.Right), int(detect.Bottom))
            cv2.rectangle(output_rgb, top_left, bottom_right, color, 2)
            confidence_text = f"{int(detect.Confidence * 100)}%"
            cv2.putText(output_rgb, f"{item} ({confidence_text})", (int(detect.Left), int(detect.Top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Dibujar bounding boxes para el modelo HSV
        for detect in detections_hsv:
            Id = detect.ClassID
            item = net_hsv.GetClassDesc(Id)
            color = (0, 255, 0) if item == "Ripe" else (0, 0, 255)
            top_left = (int(detect.Left), int(detect.Top))
            bottom_right = (int(detect.Right), int(detect.Bottom))
            cv2.rectangle(output_hsv, top_left, bottom_right, color, 2)
            confidence_text = f"{int(detect.Confidence * 100)}%"
            cv2.putText(output_hsv, f"{item} ({confidence_text})", (int(detect.Left), int(detect.Top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Mostrar ambos resultados en las ventanas
        cv2.imshow('RGB Model', output_rgb)
        cv2.imshow('HSV Model', output_hsv)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera_csi_1.release()  # Liberar la cámara CSI
    cv2.destroyAllWindows()

