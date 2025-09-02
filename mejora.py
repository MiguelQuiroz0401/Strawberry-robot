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

# Función para aplicar ecualización del histograma y umbral dinámico
def SBFilter(src):
    # Convertir la imagen RGB a LAB
    lab_image = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)

    # Ecualización del histograma en el canal L para mejorar la visibilidad
    lab_image[:, :, 0] = cv2.equalizeHist(lab_image[:, :, 0])

    # Calcular los umbrales dinámicos basados en la media de los canales
    a_channel = lab_image[:, :, 1]  # Canal 'a' para fresas maduras
    b_channel = lab_image[:, :, 2]  # Canal 'b' para fresas inmaduras

    # Umbrales dinámicos basados en la media de los canales
    mean_a = np.mean(a_channel)
    mean_b = np.mean(b_channel)

    # Ajustar los umbrales en base a la media
    ret_ripe, th_ripe = cv2.threshold(a_channel, mean_a + 15, 255, cv2.THRESH_BINARY)
    ret_unripe, th_unripe = cv2.threshold(b_channel, mean_b + 15, 255, cv2.THRESH_BINARY)

    # Filtrar pequeñas áreas no deseadas para fresas maduras
    kernel = np.ones((15, 15), np.uint8)
    filtered_ripe = cv2.morphologyEx(th_ripe, cv2.MORPH_OPEN, kernel)
    filtered_ripe = cv2.dilate(filtered_ripe, kernel, iterations=1)

    # Filtrar pequeñas áreas no deseadas para fresas inmaduras
    filtered_unripe = cv2.morphologyEx(th_unripe, cv2.MORPH_OPEN, kernel)
    filtered_unripe = cv2.dilate(filtered_unripe, kernel, iterations=1)

    # Encontrar contornos para ambas categorías (solo para análisis, no para visualización)
    contours_ripe, _ = cv2.findContours(filtered_ripe, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_unripe, _ = cv2.findContours(filtered_unripe, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours_ripe, contours_unripe

# Cargar el modelo de detección
net = jetson.inference.detectNet(
    "ssd-mobilenet-v2", 
    ["--model=models/h/ssd-mobilenet.onnx", "--labels=models/h/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes", "--batch-size=1", "--useTensorRT=true"]
)
# Inicializar las cámaras CSI con el filtro videobalance
camera1 = cv2.VideoCapture(0)#"nvarguscamerasrc sensor-id=0 awblock=true wbmode=0 exposuretimerange='200000000 200000000' ispdigitalgainrange='1 1' gainrange='10.625 10.625' ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! videobalance brightness=0.1 contrast=1.0 saturation=2.0 ! video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
camera2 = cv2.VideoCapture(1)# "nvarguscamerasrc sensor-id=1 awblock=true wbmode=0 exposuretimerange='200000000 200000000' ispdigitalgainrange='1 1' gainrange='10.625 10.625' ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! videobalance brightness=0.0 contrast=1.0 saturation=1.0 ! video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)


# Crear ventana principal para visualización
cv2.namedWindow('Detección de fresas', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detección de fresas', 960, 360)  # Ventana combinada para las dos cámaras

print('Press \'q\' to stop')

try:
    while True:
        # Capturar frames de ambas cámaras CSI
        ret1, frame1 = camera1.read()
        ret2, frame2 = camera2.read()
        
        if not ret1 or not ret2:
            print("Error al capturar frames de las cámaras.")
            break

        # Aplicar el filtro en LAB y obtener contornos para ambas cámaras
        contours_ripe_1, contours_unripe_1 = SBFilter(frame1)
        contours_ripe_2, contours_unripe_2 = SBFilter(frame2)

        # Detección en ambas imágenes usando la red
        img1 = jetson.utils.cudaFromNumpy(frame1.astype(np.float32))
        img2 = jetson.utils.cudaFromNumpy(frame2.astype(np.float32))
        
        detections1 = net.Detect(img1)
        detections1 = [d for d in detections1 if d.Confidence > confidence_threshold]
        detections1 = non_max_suppression(detections1)

        detections2 = net.Detect(img2)
        detections2 = [d for d in detections2 if d.Confidence > confidence_threshold]
        detections2 = non_max_suppression(detections2)

        # Dibujar las detecciones en ambas imágenes
        output_frame1 = frame1.copy()
        output_frame2 = frame2.copy()

        # Función auxiliar para dibujar detecciones
        def draw_detections(detections, frame):
            for detect in detections:
                Id = detect.ClassID
                item = net.GetClassDesc(Id)
                
                if item == "Ripe":
                    color = (0, 255, 0)  # Verde para maduras
                elif item == "Unripe":
                    color = (0, 165, 255)  # Naranja para inmaduras
                else:
                    color = (0, 0, 255)  # Rojo para desconocidas

                top_left = (int(detect.Left), int(detect.Top))
                bottom_right = (int(detect.Right), int(detect.Bottom))

                cv2.rectangle(frame, top_left, bottom_right, color, 3)
                confidence_text = f"{int(detect.Confidence * 100)}%"
                cv2.putText(frame, f"{item} ({confidence_text})", (int(detect.Left), int(detect.Top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Dibujar detecciones en ambos frames
        draw_detections(detections1, output_frame1)
        draw_detections(detections2, output_frame2)

        # Concatenar ambas imágenes horizontalmente
        combined_frame = np.hstack((output_frame1, output_frame2))

        # Mostrar la imagen combinada en la ventana 'Detección de fresas'
        cv2.imshow('Detección de fresas', combined_frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    # Liberar recursos
    camera1.release()
    camera2.release()
    cv2.destroyAllWindows()

