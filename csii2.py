import cv2
import numpy as np
import jetson.inference
import jetson.utils
import freenect

# Definir umbral de confianza
confidence_threshold = 0.5

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
camera_csi_1 = cv2.VideoCapture("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
camera_csi_2 = cv2.VideoCapture("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

# Función para capturar el frame RGB del Kinect
def get_kinect_rgb():
    frame, _ = freenect.sync_get_video()
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# Función para capturar el frame de profundidad del Kinect
def get_kinect_depth():
    depth, _ = freenect.sync_get_depth()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(depth.astype(np.uint8), cv2.COLORMAP_JET)

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

def nothing(x):
    pass

# Crear ventana principal para visualización
cv2.namedWindow('Video')
cv2.moveWindow('Video', 5, 5)
cv2.resizeWindow('Video', 1280, 720)

# Crear ventana para sliders
cv2.namedWindow('Controls')
cv2.moveWindow('Controls', 1350, 5)

# Definir kernel y cargar imagen
kernel = np.ones((5, 5), np.uint8)
imn = cv2.imread('blank.bmp')

# Crear sliders en la ventana de controles
cv2.createTrackbar('val1', 'Controls', 37, 1000, nothing)
cv2.createTrackbar('val2', 'Controls', 43, 1000, nothing)
cv2.createTrackbar('bin', 'Controls', 20, 50, nothing)
cv2.createTrackbar('erode', 'Controls', 4, 10, nothing)
cv2.createTrackbar('epsilon', 'Controls', 1, 100, nothing)
cv2.createTrackbar('spacing', 'Controls', 30, 100, nothing)

def pretty_depth(depth):
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

def RegionCheck(foo, ListPath):
    if (foo <= 130) and (ListPath[0] != 0):
        ListPath[0] = 0
    if (foo > 130) and (foo <= 320) and (ListPath[1] != 0):
        ListPath[1] = 0
    if (foo > 320) and (foo <= 510) and (ListPath[2] != 0):
        ListPath[2] = 0
    if (foo > 510) and (ListPath[3] != 0):
        ListPath[3] = 0
    return ListPath

def imgshow(ListPath, t, imn, Winname):
    if ListPath[1:3] == [1, 1]:
        imn = cv2.imread(f"{t}frwd.bmp")
    elif ListPath[2:4] == [1, 1]:
        imn = cv2.imread(f"{t}right.bmp")
    elif ListPath[0:2] == [1, 1]:
        imn = cv2.imread(f"{t}left.bmp")
    else:
        imn = cv2.imread(f"{t}back.bmp")
    cv2.imshow(Winname, imn)

print('Press \'b\' in window to stop')

# Inicializar flags
flag120 = [1, 1, 1, 1]
flag140 = [1, 1, 1, 1]
f8 = f10 = f12 = f14 = 0

try:
    while True:
        # Capturar frames de las cámaras CSI
        ret1, frame_csi_1 = camera_csi_1.read()
        ret2, frame_csi_2 = camera_csi_2.read()
        if not ret1 or not ret2:
            print("Error al capturar frames de las cámaras CSI.")
            break

        # Capturar frames del Kinect
        frame_kinect_rgb = get_kinect_rgb()
        frame_kinect_depth = get_kinect_depth()

        # Aplicar el modelo de detección en las cámaras CSI
        frames = [frame_csi_1, frame_csi_2]
        processed_frames = []

        for frame in frames:
            img = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float))
            detections = net.Detect(img)
            detections = [d for d in detections if d.Confidence > confidence_threshold]
            detections = non_max_suppression(detections)

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

            processed_frames.append(output_frame)

        # Redimensionar frames
        frame_kinect_rgb = cv2.resize(frame_kinect_rgb, (640, 480))
        frame_kinect_depth = cv2.resize(frame_kinect_depth, (640, 480))
        processed_frames[0] = cv2.resize(processed_frames[0], (640, 480))
        processed_frames[1] = cv2.resize(processed_frames[1], (640, 480))

        # Combinar imágenes de las cuatro cámaras
        top_row = np.hstack((processed_frames[0], processed_frames[1]))
        bottom_row = np.hstack((frame_kinect_rgb, frame_kinect_depth))
        combined_frame = np.vstack((top_row, bottom_row))

        # Redimensionar la imagen combinada para ajustarse a la ventana 'Video'
        combined_frame = cv2.resize(combined_frame, (1280, 720))

        # Mostrar la imagen combinada en la ventana 'Video'
        cv2.imshow('Video', combined_frame)

        # Procesar la cámara de profundidad del Kinect
        dst = pretty_depth(freenect.sync_get_depth()[0])
        cv2.flip(dst, 0, dst)
        cv2.flip(dst, 0, dst)
        
        # Dibujar un rectángulo en la imagen de profundidad
        cv2.rectangle(dst, (0, 0), (640, 480), (40, 100, 0), 2)
        
        # Obtener posiciones de los trackbars
        binn = cv2.getTrackbarPos('bin', 'Controls')
        e = cv2.getTrackbarPos('erode', 'Controls')
        dst = (dst // binn) * binn
        dst = cv2.erode(dst, kernel, iterations=e)
        
        v1 = cv2.getTrackbarPos('val1', 'Controls')
        v2 = cv2.getTrackbarPos('val2', 'Controls')
        edges = cv2.Canny(dst, v1, v2)
        
        # Encontrar contornos y dibujarlos
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dst, contours, -1, (0, 0, 255), -1)
        
        spac = cv2.getTrackbarPos('spacing', 'Controls')
        rows, cols = dst.shape

        # Iterar a través de los puntos de la cuadrícula
        for i in range(0, rows, spac):
            for j in range(0, cols, spac):
                cv2.circle(dst, (j, i), 1, (0, 255, 0), 1)
                depth_value = dst[i, j]
                if depth_value == 80:
                    f8 = 1
                    cv2.putText(dst, "0", (j, i), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 20), 2)
                    cv2.putText(dst, "Collision Alert!", (30, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, 2, 1)
                    imn = cv2.imread("Collision Alert.bmp")
                    cv2.imshow('Navig', imn)
                elif depth_value == 100:
                    f10 = 1
                    cv2.putText(dst, "1", (j, i), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 20), 2)
                    cv2.putText(dst, "Very Close proximity. Reverse", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, 2, 1)
                    if not f8:
                        imn = cv2.imread("VCP Reverse.bmp")
                        cv2.imshow('Navig', imn)
                elif depth_value == 120:
                    f12 = 1
                    cv2.putText(dst, "2", (j, i), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 20), 2)
                    flag120 = RegionCheck(j, flag120)
                    if not f8 and not f10:
                        imgshow(flag120, 120, imn, 'Navig')
                elif depth_value == 140:
                    f14 = 1
                    cv2.putText(dst, "3", (j, i), cv2.FONT_HERSHEY_PLAIN, 1, (0, 200, 20), 1)
                    flag140 = RegionCheck(j, flag140)
                    if not f8 and not f10 and not f12:
                        imgshow(flag140, 140, imn, 'Navig')

        # Dibujar líneas en la imagen de profundidad
        cv2.line(dst, (130, 0), (130, 480), 0, 1)
        cv2.line(dst, (320, 0), (320, 480), 0, 1)
        cv2.line(dst, (510, 0), (510, 480), 0, 1)
        cv2.imshow('Navig', dst)
        
        # Salir del bucle al presionar 'b'
        if cv2.waitKey(1) & 0xFF == ord('b'):
            break
finally:
    # Asegurarse de cerrar todas las ventanas y liberar los recursos
    camera_csi_1.release()
    camera_csi_2.release()
    cv2.destroyAllWindows()

