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

# Inicializar las cámaras CSI
camera_csi_1 = cv2.VideoCapture("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)320, height=(int)240, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
camera_csi_2 = cv2.VideoCapture("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)320, height=(int)240, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

# Crear ventana principal para visualización
cv2.namedWindow('Video')
cv2.moveWindow('Video', 5, 5)
cv2.resizeWindow('Video', 640, 240)

print('Press \'q\' to stop')

try:
    while True:
        # Capturar frames de las cámaras CSI
        ret1, frame_csi_1 = camera_csi_1.read()
        ret2, frame_csi_2 = camera_csi_2.read()
        if not ret1 or not ret2:
            print("Error al capturar frames de las cámaras CSI.")
            break

        # Medir el tiempo de inicio de la detección
        start_time = time.time()

        # Aplicar el modelo de detección en las cámaras CSI
        frames = [frame_csi_1, frame_csi_2]
        processed_frames = []

        for frame in frames:
            img = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float))
            detections = net.Detect(img)
            detections = [d for d in detections if d.Confidence > confidence_threshold]
            detections = non_max_suppression(detections)

            # Medir el tiempo de detección
            detection_time = time.time() - start_time

            # Crear una imagen de salida para dibujar las detecciones
            output_frame = frame.copy()

            # Dibujar bounding boxes
            for detect in detections:
                Id = detect.ClassID
                item = net.GetClassDesc(Id)
                color = (0, 255, 0) if item == "Ripe" else (0, 0, 255)
                top_left = (int(detect.Left), int(detect.Top))
                bottom_right = (int(detect.Right), int(detect.Bottom))

                overlay = output_frame.copy()
                cv2.rectangle(overlay, top_left, bottom_right, color, thickness=cv2.FILLED)
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
                cv2.rectangle(overlay, top_left, bottom_right, color, 2)
                cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
                confidence_text = f"{int(detect.Confidence * 100)}%"
                cv2.putText(output_frame, f"{item} ({confidence_text})", (int(detect.Left), int(detect.Top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Mostrar tiempo de detección en la imagen
                detection_text = f"Detección: {detection_time:.2f} s"
                cv2.putText(output_frame, detection_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            processed_frames.append(output_frame)

        # Redimensionar frames a 320x240 sin cambiar la relación de aspecto
        processed_frames[0] = cv2.resize(processed_frames[0], (320, 240))
        processed_frames[1] = cv2.resize(processed_frames[1], (320, 240))

        # Combinar imágenes de las dos cámaras en una fila horizontal
        combined_frame = np.hstack((processed_frames[0], processed_frames[1]))

        # Mantener la ventana combinada en 640x240
        combined_frame = cv2.resize(combined_frame, (640, 240))

        # Mostrar la imagen combinada en la ventana 'Video'
        cv2.imshow('Video', combined_frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    camera_csi_1.release()
    camera_csi_2.release()
    cv2.destroyAllWindows()

