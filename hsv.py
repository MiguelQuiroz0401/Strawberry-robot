import cv2
import numpy as np

# Inicialización de la cámara USB
camera_usb = cv2.VideoCapture(0)  # Cambiar al índice de tu cámara USB si es necesario
camera_usb.set(3, 640)  # Ancho de la ventana de visualización
camera_usb.set(4, 480)  # Alto de la ventana de visualización

# Función para actualizar la imagen cuando se cambian los valores de los sliders
def nothing(x):
    pass

# Crear ventana para los sliders de calibración
cv2.namedWindow("Calibración HSV")

# Crear sliders para ajustar los valores HSV
cv2.createTrackbar("H min", "Calibración HSV", 0, 179, nothing)
cv2.createTrackbar("S min", "Calibración HSV", 120, 255, nothing)
cv2.createTrackbar("V min", "Calibración HSV", 70, 255, nothing)
cv2.createTrackbar("H max", "Calibración HSV", 10, 179, nothing)
cv2.createTrackbar("S max", "Calibración HSV", 255, 255, nothing)
cv2.createTrackbar("V max", "Calibración HSV", 255, 255, nothing)

while True:
    # Captura un fotograma de la cámara
    ret, frame = camera_usb.read()
    if not ret:
        print("No se puede acceder a la cámara")
        break

    # Obtener los valores de los sliders
    h_min = cv2.getTrackbarPos("H min", "Calibración HSV")
    s_min = cv2.getTrackbarPos("S min", "Calibración HSV")
    v_min = cv2.getTrackbarPos("V min", "Calibración HSV")
    h_max = cv2.getTrackbarPos("H max", "Calibración HSV")
    s_max = cv2.getTrackbarPos("S max", "Calibración HSV")
    v_max = cv2.getTrackbarPos("V max", "Calibración HSV")

    # Definir el rango HSV usando los valores de los sliders
    lower_red = np.array([h_min, s_min, v_min])
    upper_red = np.array([h_max, s_max, v_max])

    # Convertir la imagen de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Crear una máscara para detectar los rojos
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Mostrar la imagen de resultados con solo las fresas rojas visibles
    cv2.imshow("Fresas Detectadas", result)
    
    # Mostrar el fotograma original
    cv2.imshow("Original", frame)

    # Espera la tecla 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra las ventanas
camera_usb.release()
cv2.destroyAllWindows()

