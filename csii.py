import cv2
import numpy as np
import jetson.inference
import jetson.utils
import freenect

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
camera_csi_1 = cv2.VideoCapture("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
camera_csi_2 = cv2.VideoCapture("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

# Umbral de confianza para filtrado
confidence_threshold = 0.5

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

# Crear ventanas para los trackbars
cv2.namedWindow('Video')
cv2.moveWindow('Video', 5, 5)
cv2.namedWindow('Navig', cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow('Navig', 400, 100)
cv2.moveWindow('Navig', 700, 5)

# Definir kernel y cargar imagen
kernel = np.ones((5, 5), np.uint8)
imn = cv2.imread('blank.bmp')

# Crear trackbars
cv2.createTrackbar('val1', 'Video', 37, 1000, nothing)
cv2.createTrackbar('val2', 'Video', 43, 1000, nothing)
cv2.createTrackbar('bin', 'Video', 20, 50, nothing)
cv2.createTrackbar('erode', 'Video', 4, 10, nothing)
cv2.createTrackbar('epsilon', 'Video', 1, 100, nothing)
cv2.createTrackbar('spacing', 'Video', 30, 100, nothing)

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
    frame_kinect_rgb = cv2.resize(frame_kinect_rgb, (320, 240))
    frame_kinect_depth = cv2.resize(frame_kinect_depth, (320, 240))
    frame_csi_1 = cv2.resize(processed_frames[0], (320, 240))
    frame_csi_2 = cv2.resize(processed_frames[1], (320, 240))

    # Mostrar frames combinados en una ventana
    combined_frame = np.hstack((np.vstack((frame_csi_1, frame_csi_2)), np.vstack((frame_kinect_rgb, frame_kinect_depth))))
    cv2.imshow('Video', combined_frame)

    # Mostrar la cámara de profundidad en una ventana separada
    cv2.imshow('Depth', frame_kinect_depth)

    key = cv2.waitKey(1)
    if key == ord('b'):
        break

camera_csi_1.release()
camera_csi_2.release()
cv2.destroyAllWindows()
