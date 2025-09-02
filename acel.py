import cv2
import numpy as np
import jetson.inference
import jetson.utils
import time
import re
from dronekit import connect, VehicleMode
import serial
import threading

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

# Conectar al Pixhawk
vehicle = connect('/dev/ttyTHS1', baud=57600, wait_ready=True, timeout=60)
print("Pixhawk conectado.")

# Conectar al ESP32
esp32 = serial.Serial('/dev/ttyUSB0', 115200)
time.sleep(2)
print("ESP32 conectado.")

# Cambiar al modo MANUAL (Modo manual por defecto)
vehicle.mode = VehicleMode("MANUAL")
while not vehicle.mode.name == 'MANUAL':
    time.sleep(1)
print("Vehículo ahora en modo MANUAL.")

# Configuración del canal de dirección y PWM
steering_channel = 3
neutral_pwm = 1500
turn_left_pwm = 2000
turn_right_pwm = 850
distance_threshold = 45

# Tiempo de espera para mantener el giro
hold_turn_time = 0.1
last_turn_time = 0

# Inicializar distancias de los sensores
distances = {1: float('inf'), 2: float('inf'), 3: float('inf'), 4: float('inf'), 5: float('inf'), 6: float('inf')}

# Cargar el modelo de detección
net = jetson.inference.detectNet("ssd-mobilenet-v2", [
    "--model=models/fresas100/ssd-mobilenet.onnx",
    "--labels=models/fresas100/labels.txt",
    "--input-blob=input_0",
    "--output-cvg=scores",
    "--output-bbox=boxes"
])

# Inicializar las cámaras CSI
camera_csi_1 = cv2.VideoCapture("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)320, height=(int)240, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
camera_csi_2 = cv2.VideoCapture("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)320, height=(int)240, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

# Crear ventana para visualización
cv2.namedWindow('Video')
cv2.moveWindow('Video', 5, 5)
cv2.resizeWindow('Video', 640, 240)

# Función para manejar la detección de fresas (Ejecutada en un hilo separado)
def detection_thread():
    while True:
        ret1, frame_csi_1 = camera_csi_1.read()
        ret2, frame_csi_2 = camera_csi_2.read()
        if not ret1 or not ret2:
            print("Error al capturar frames de las cámaras.")
            break

        frames = [frame_csi_1, frame_csi_2]
        processed_frames = []

        for frame in frames:
            img = jetson.utils.cudaFromNumpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA).astype(np.float))
            detections = net.Detect(img)
            detections = [d for d in detections if d.Confidence > confidence_threshold]
            detections = non_max_suppression(detections)

            output_frame = frame.copy()
            for detect in detections:
                cv2.rectangle(output_frame, (int(detect.Left), int(detect.Top)), 
                              (int(detect.Right), int(detect.Bottom)), (0, 255, 0), 2)
            
            processed_frames.append(output_frame)

        if processed_frames:
            combined_frame = np.concatenate(processed_frames, axis=1)
            cv2.imshow('Video', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Función para manejar los sensores y la reacción del vehículo (Ejecutada en otro hilo)
def sensors_and_control_thread():
    global last_turn_time
    while True:
        if esp32.in_waiting > 0:
            distance_data = esp32.readline().decode().strip()
            match = re.match(r'Sensor (\d+): ([\d.]+) cm', distance_data)
            if match:
                sensor_id = int(match.group(1))
                distance = float(match.group(2))
                distances[sensor_id] = distance

                current_time = time.time()

                left_distance = min(distances[4], distances[5], distances[6])
                right_distance = min(distances[1], distances[2], distances[3])

                if left_distance < distance_threshold and right_distance < distance_threshold:
                    # Cambiar a ACRO para esquivar obstáculos
                    vehicle.mode = VehicleMode("ACRO")
                    while not vehicle.mode.name == 'ACRO':
                        time.sleep(1)

                    if left_distance < right_distance:
                        vehicle.channels.overrides[steering_channel] = turn_right_pwm
                    else:
                        vehicle.channels.overrides[steering_channel] = turn_left_pwm
                    last_turn_time = current_time

                elif left_distance < distance_threshold:
                    vehicle.mode = VehicleMode("ACRO")
                    while not vehicle.mode.name == 'ACRO':
                        time.sleep(1)

                    vehicle.channels.overrides[steering_channel] = turn_right_pwm
                    last_turn_time = current_time

                elif right_distance < distance_threshold:
                    vehicle.mode = VehicleMode("ACRO")
                    while not vehicle.mode.name == 'ACRO':
                        time.sleep(1)

                    vehicle.channels.overrides[steering_channel] = turn_left_pwm
                    last_turn_time = current_time

                elif current_time - last_turn_time > hold_turn_time:
                    # Volver al modo MANUAL después de evitar el obstáculo
                    vehicle.mode = VehicleMode("MANUAL")
                    while not vehicle.mode.name == 'MANUAL':
                        time.sleep(1)
                    vehicle.channels.overrides[steering_channel] = neutral_pwm

# Crear los hilos para la detección y el control
detection_thread = threading.Thread(target=detection_thread)
sensors_and_control_thread = threading.Thread(target=sensors_and_control_thread)

# Iniciar los hilos
detection_thread.start()
sensors_and_control_thread.start()

# Esperar a que los hilos terminen
detection_thread.join()
sensors_and_control_thread.join()

# Cerrar recursos
vehicle.close()
esp32.close()
camera_csi_1.release()
camera_csi_2.release()
cv2.destroyAllWindows()
