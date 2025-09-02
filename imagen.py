import cv2
import numpy as np
import jetson.inference
import jetson.utils

def detectar_fresas(img_bgr, lower_color1, upper_color1, lower_color2, upper_color2):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask_color1 = cv2.inRange(img_hsv, lower_color1, upper_color1)
    mask_color2 = cv2.inRange(img_hsv, lower_color2, upper_color2)
    mask_fresas_hsv = cv2.bitwise_or(mask_color1, mask_color2)
    
    # Aplicar operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)  # Definir un kernel de 5x5
    mask_fresas_hsv = cv2.morphologyEx(mask_fresas_hsv, cv2.MORPH_OPEN, kernel)  # Apertura
    mask_fresas_hsv = cv2.morphologyEx(mask_fresas_hsv, cv2.MORPH_CLOSE, kernel)  # Cierre

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

# Cargar una imagen de prueba desde archivo
img_path = "/home/miguel/450_1000.jpg"  # Ruta de la imagen

# Crear una ventana para la interfaz de ajuste de HSV
cv2.namedWindow('HSV Slider Controls')

# Crear sliders para controlar los valores HSV
# Rojos
cv2.createTrackbar('Lower Red H', 'HSV Slider Controls', 0, 180, lambda x: None)
cv2.createTrackbar('Lower Red S', 'HSV Slider Controls', 100, 255, lambda x: None)
cv2.createTrackbar('Lower Red V', 'HSV Slider Controls', 50, 255, lambda x: None)
cv2.createTrackbar('Upper Red H', 'HSV Slider Controls', 10, 180, lambda x: None)
cv2.createTrackbar('Upper Red S', 'HSV Slider Controls', 255, 255, lambda x: None)
cv2.createTrackbar('Upper Red V', 'HSV Slider Controls', 255, 255, lambda x: None)

# Verdes/amarillo-verde
cv2.createTrackbar('Lower Green H', 'HSV Slider Controls', 25, 180, lambda x: None)
cv2.createTrackbar('Lower Green S', 'HSV Slider Controls', 80, 255, lambda x: None)
cv2.createTrackbar('Lower Green V', 'HSV Slider Controls', 80, 255, lambda x: None)
cv2.createTrackbar('Upper Green H', 'HSV Slider Controls', 35, 180, lambda x: None)
cv2.createTrackbar('Upper Green S', 'HSV Slider Controls', 255, 255, lambda x: None)
cv2.createTrackbar('Upper Green V', 'HSV Slider Controls', 255, 255, lambda x: None)

print("Presiona 'q' para salir")

try:
    while True:
        # Leer la imagen desde archivo
        frame = cv2.imread(img_path)
        
        if frame is None:
            print("Error al cargar la imagen.")
            break

        # Obtener los valores actuales de los sliders
        lower_red1 = np.array([cv2.getTrackbarPos('Lower Red H', 'HSV Slider Controls'), 
                               cv2.getTrackbarPos('Lower Red S', 'HSV Slider Controls'),
                               cv2.getTrackbarPos('Lower Red V', 'HSV Slider Controls')])

        upper_red1 = np.array([cv2.getTrackbarPos('Upper Red H', 'HSV Slider Controls'), 
                               cv2.getTrackbarPos('Upper Red S', 'HSV Slider Controls'),
                               cv2.getTrackbarPos('Upper Red V', 'HSV Slider Controls')])

        lower_yellowgreen = np.array([cv2.getTrackbarPos('Lower Green H', 'HSV Slider Controls'), 
                                       cv2.getTrackbarPos('Lower Green S', 'HSV Slider Controls'),
                                       cv2.getTrackbarPos('Lower Green V', 'HSV Slider Controls')])

        upper_yellowgreen = np.array([cv2.getTrackbarPos('Upper Green H', 'HSV Slider Controls'), 
                                       cv2.getTrackbarPos('Upper Green S', 'HSV Slider Controls'),
                                       cv2.getTrackbarPos('Upper Green V', 'HSV Slider Controls')])

        # Procesar la imagen con los valores actuales de HSV
        hsv = detectar_fresas(frame, lower_red1, upper_red1, lower_yellowgreen, upper_yellowgreen)
        
        img = jetson.utils.cudaFromNumpy(hsv.astype(np.float32))
        
        detections = net.Detect(img)

        draw_detections(detections, frame, net)

        # Mostrar la imagen y la detección de fresas
        cv2.imshow('Detección de fresas', frame)
        cv2.imshow('Imagen HSV', hsv)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    cv2.destroyAllWindows()

