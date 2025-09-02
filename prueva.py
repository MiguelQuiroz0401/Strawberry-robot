import cv2
import numpy as np
import jetson.inference
import jetson.utils

def detectar_fresas(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    lower_yellowgreen = np.array([25, 80, 80])
    upper_yellowgreen = np.array([35, 255, 255])
    mask_yellowgreen = cv2.inRange(img_hsv, lower_yellowgreen, upper_yellowgreen)

    mask_fresas_hsv = cv2.bitwise_or(mask_red, mask_yellowgreen)
    fresas_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask=mask_fresas_hsv)

    return fresas_hsv

def draw_detections(detections, frame, net):
    for detect in detections:
        Id = detect.ClassID
        item = net.GetClassDesc(Id)
        color = (0, 255, 0) if item == "Ripe" else (0, 165, 255) if item == "Unripe" else (0, 0, 255)
        top_left = (int(detect.Left), int(detect.Top))
        bottom_right = (int(detect.Right), int(detect.Bottom))
        cv2.rectangle(frame, top_left, bottom_right, color, 3)
        confidence_text = f"{int(detect.Confidence * 100)}%"
        cv2.putText(frame, f"{item} ({confidence_text})", (int(detect.Left), int(detect.Top) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

net = jetson.inference.detectNet("ssd-mobilenet-v2", 
    ["--model=models/h/ssd-mobilenet.onnx", "--labels=models/h/labels.txt", "--input-blob=input_0", 
     "--output-cvg=scores", "--output-bbox=boxes", "--batch-size=1", "--useTensorRT=true"])

camera1 = cv2.VideoCapture(0)
camera2 = cv2.VideoCapture(1)

cv2.namedWindow('Detección de fresas', cv2.WINDOW_NORMAL)
cv2.namedWindow('Imagen HSV', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detección de fresas', 960, 360)
cv2.resizeWindow('Imagen HSV', 960, 360)

print("Presiona 'q' para salir")

try:
    while True:
        ret1, frame1 = camera1.read()
        ret2, frame2 = camera2.read()
        
        if not ret1 or not ret2:
            print("Error al capturar frames de las cámaras.")
            break

        hsv1 = detectar_fresas(frame1)
        hsv2 = detectar_fresas(frame2)
        
        img1 = jetson.utils.cudaFromNumpy(hsv1.astype(np.float32))
        img2 = jetson.utils.cudaFromNumpy(hsv2.astype(np.float32))
        
        detections1 = net.Detect(img1)
        detections2 = net.Detect(img2)

        draw_detections(detections1, frame1, net)
        draw_detections(detections2, frame2, net)

        combined_frame = np.hstack((frame1, frame2))
        combined_hsv = np.hstack((hsv1, hsv2))
        
        cv2.imshow('Detección de fresas', combined_frame)
        cv2.imshow('Imagen HSV', combined_hsv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    camera1.release()
    camera2.release()
    cv2.destroyAllWindows()
