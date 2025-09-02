import jetson.inference
import jetson.utils
import cv2
import numpy as np

# Cargar el modelo detectNet (por defecto usa SSD-Mobilenet)
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Abrir la webcam (ajusta el índice si es necesario)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("? Error: No se pudo abrir la cámara.")
    exit()

while True:
    # Capturar el frame
    ret, frame = cap.read()
    if not ret:
        print("? Error: No se pudo capturar el frame.")
        break

    # Convertir a imagen CUDA
    img_cuda = jetson.utils.cudaFromNumpy(frame)

    # Detectar objetos
    detections = net.Detect(img_cuda, frame.shape[1], frame.shape[0])

    # Dibujar las detecciones
    for detection in detections:
        x1, y1, x2, y2 = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
        confidence = detection.Confidence

        # Dibujar el rectángulo y la confianza
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{net.GetClassDesc(detection.ClassID)} ({confidence:.2f})", 
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el frame
    cv2.imshow("DetectNet - Jetson Nano", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

