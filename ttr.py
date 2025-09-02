import jetson_inference
import jetson_utils
import cv2
import numpy as np

# Cargar el modelo ONNX
model_path = "models/fresas/ssd-mobilenet.onnx"
labels_path = "models/fresas/labels.txt"
net = jetson_inference.detectNet(argv=['--model=' + model_path, '--labels=' + labels_path, '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'])

# Inicializar la cámara (usando OpenCV para capturar frames)
cap = cv2.VideoCapture(0)  # Usa la cámara en /dev/video0

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

while True:
    # Capturar frame de la cámara
    ret, frame = cap.read()
    
    if not ret:
        print("Error: No se pudo capturar un frame.")
        break
    
    # Convertir la imagen a formato CUDA para usarla con Jetson Inference
    cuda_frame = jetson_utils.cudaFromNumpy(frame)
    
    # Realizar detección
    detections = net.Detect(cuda_frame)

    # Verificar detección de fresas maduras
    ripe_detected = False
    for detection in detections:
        if net.GetClassDesc(detection.ClassID) == "ripe":
            ripe_detected = True
            print("Fresa madura detectada")
        
        # Dibujar la caja de detección y la etiqueta
        left, top, right, bottom = int(detection.Left), int(detection.Top), int(detection.Right), int(detection.Bottom)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        label = f"{net.GetClassDesc(detection.ClassID)}: {detection.Confidence:.2f}"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Mostrar el frame con detecciones
    cv2.imshow("Detecciones", frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()

