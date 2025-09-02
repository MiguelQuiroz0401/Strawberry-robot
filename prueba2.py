import cv2
import freenect
import numpy as np

def get_kinect_frames():
    # Captura de la cámara Kinect RGB
    rgb, _ = freenect.sync_get_video()
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    
    # Captura de la cámara Kinect Depth
    depth, _ = freenect.sync_get_depth()
    depth = np.uint8(depth / np.max(depth) * 255)  # Normalización
    depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

    return rgb, depth

def main():
    # Abre la webcam
    webcam = cv2.VideoCapture(0)

    if not webcam.isOpened():
        print("Error: No se puede abrir la webcam.")
        return

    while True:
        # Captura de la webcam
        ret, frame_webcam = webcam.read()
        if not ret:
            print("Error: No se puede capturar la imagen de la webcam.")
            break

        # Captura de las imágenes de la Kinect
        frame_kinect_rgb, frame_kinect_depth = get_kinect_frames()

        # Muestra las imágenes
        cv2.imshow("Webcam", frame_webcam)
        cv2.imshow("Kinect RGB", frame_kinect_rgb)
        cv2.imshow("Kinect Depth", frame_kinect_depth)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libera los recursos
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

