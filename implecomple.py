import cv2
import numpy as np
import jetson.inference
import jetson.utils
import time
from dronekit import connect, VehicleMode
from pymavlink import mavutil
import serial
import threading
import math
from datetime import datetime
from collections import deque


lat_values = []
lon_values = []
last_save_time = 0

confidence_threshold = 0.5
vehicle = None
is_vehicle_connected = False

# Parámetros de distancia de los sensores
min_distance = 3     # Distancia mínima (en cm)
max_distance = 250   # Distancia máxima (en cm)
distances = [500, 500, 500, 500, 500, 500]  # Distancias de los 6 sensores ultrasónicos iniciales (en cm)

# Configura el puerto serial al que está conectado el ESP32
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

# Modo previo del vehículo antes de cambiar a MANUAL
previous_mode = None

# Conexión al vehículo
def vehicle_connect():
    global vehicle, is_vehicle_connected

    if vehicle is None:
        try:
            vehicle = connect('/dev/ttyTHS1', baud=57600, wait_ready=True, timeout=60)
        except:
            print('Error de conexión. Intentando de nuevo...')
            vehicle = connect('/dev/ttyUSB1', wait_ready=True, baud=921600)
            time.sleep(1)

    if vehicle is None:
        is_vehicle_connected = False
        return False
    else:
        is_vehicle_connected = True
        return True

# Función para enviar valores a los canales RC
def send_rc_control(channel_1_value, channel_3_value):
    if is_vehicle_connected:
        msg = vehicle.message_factory.rc_channels_override_encode(
            0,     # target_system (0 para el sistema actual)
            0,     # target_component (0 para el componente actual)
            channel_1_value,  # Canal 1 (Throttle)
            0,     # Canal 2
            channel_3_value,  # Canal 3 (Steering)
            0,     # Canal 4
            0,     # Canal 5
            0,     # Canal 6
            0,     # Canal 7
            0      # Canal 8
        )
        vehicle.send_mavlink(msg)
        vehicle.flush()
        print(f"Enviado - Throttle: {channel_1_value}, Steering: {channel_3_value}")

# Función para cambiar de modo del vehículo
def change_vehicle_mode(mode):
    global previous_mode

    if not is_vehicle_connected:
        print("El vehículo no está conectado")
        return

    # Si el modo actual no es el deseado, cambiarlo
    if vehicle.mode.name != mode:
        # Guardar el modo previo solo si se va a cambiar a MANUAL
        if mode == 'MANUAL' and previous_mode is None:
            previous_mode = vehicle.mode.name

        # Cambiar el modo
        vehicle.mode = VehicleMode(mode)
        while vehicle.mode.name != mode:
            time.sleep(0.001)
        print(f"Modo cambiado a {mode}")

# Función para leer los datos desde el puerto serial (ESP32)
def read_sensor_data():
    try:
        line = ser.readline().decode('utf-8').strip()  # Leer una línea y eliminar saltos de línea
        if line:
            distances_list = line.split(',')
            if len(distances_list) == 6:
                distances_list = [float(distance) for distance in distances_list]
                return distances_list
            else:
                print("Error en los datos recibidos: El número de sensores no es correcto.")
                return [500, 500, 500, 500, 500, 500]
        else:
            print("Error: La línea recibida está vacía.")
            return [500, 500, 500, 500, 500, 500]  # Si la línea está vacía, retornamos valores por defecto
    except Exception as e:
        print(f"Error al leer los datos del sensor: {e}")
        return [500, 500, 500, 500, 500, 500]  # Si ocurre un error, retornamos valores por defecto

# Función para decidir la maniobra basada en los sensores
def maneuver_based_on_sensors():
    global previous_mode

    # Leer las distancias de los sensores ultrasónicos desde el ESP32
    distances = read_sensor_data()
    print(f"Distancias: {distances}")
    show_radar_visualization(distances)
    # Evaluar los sensores de la izquierda y derecha
    left_sensors = distances[0:3]  # Sensores 1, 2, 3
    right_sensors = distances[3:6]  # Sensores 4, 5, 6

    # Detectar el mínimo en ambos lados
    min_left = min(left_sensors)
    min_right = min(right_sensors)

    # Si detecta obstáculo, elige el lado con menor distancia
    if min_left < 30 or min_right < 30:
        change_vehicle_mode('MANUAL')
        
        if min_left < min_right:
            print("Obstáculo detectado a la izquierda, maniobrando a la derecha.")
            send_rc_control(2000, 2000)  # Aumentar ligeramente la velocidad hacia adelante y girar suavemente a la derecha
        else:
            print("Obstáculo detectado a la derecha, maniobrando a la izquierda.")
            send_rc_control(2000, 900)  # Aumentar ligeramente la velocidad hacia adelante y girar suavemente a la izquierda

    else:
        change_vehicle_mode('AUTO')
        send_rc_control(1500, 1500)  # Detener el movimiento y enderezar dirección

# Función que maneja la ejecución concurrente
def sensor_thread(stop_event):
    while not stop_event.is_set():
        maneuver_based_on_sensors()
        time.sleep(0.001)


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

# Función para aplicar filtrado en el espacio LAB (sin visualización)
def SBFilter(src):
    # Convertir la imagen RGB a LAB
    lab_image = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    a_channel = lab_image[:, :, 1]  # Canal 'a' para resaltar fresas maduras
    b_channel = lab_image[:, :, 2]  # Canal 'b' para resaltar fresas inmaduras

    # Ajusta el umbral para una mejor segmentación de fresas maduras
    ret3, th2_ripe = cv2.threshold(a_channel, 160, 255, cv2.THRESH_BINARY)

    # Ajusta el umbral para fresas inmaduras
    ret3, th2_unripe = cv2.threshold(b_channel, 140, 255, cv2.THRESH_BINARY)

    # Filtrar pequeñas áreas no deseadas para fresas maduras
    kernel = np.ones((15, 15), np.uint8)
    filtered_ripe = cv2.morphologyEx(th2_ripe, cv2.MORPH_OPEN, kernel)
    filtered_ripe = cv2.dilate(filtered_ripe, kernel, iterations=1)

    # Filtrar pequeñas áreas no deseadas para fresas inmaduras
    filtered_unripe = cv2.morphologyEx(th2_unripe, cv2.MORPH_OPEN, kernel)
    filtered_unripe = cv2.dilate(filtered_unripe, kernel, iterations=1)

    # Encontrar contornos para ambas categorías (solo para análisis, no para visualización)
    contours_ripe, _ = cv2.findContours(filtered_ripe, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours_unripe, _ = cv2.findContours(filtered_unripe, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours_ripe, contours_unripe



#main 
# Cargar el modelo de detección
net = jetson.inference.detectNet(
    "ssd-mobilenet-v2", 
    ["--model=models/fresas100/ssd-mobilenet.onnx", "--labels=models/fresas100/labels.txt", "--input-blob=input_0", "--output-cvg=scores", "--output-bbox=boxes", "--batch-size=1", "--useTensorRT=true"]
)

# Inicializar las cámaras CSI
camera1 = cv2.VideoCapture("nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)
camera2 = cv2.VideoCapture("nvarguscamerasrc sensor-id=1 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)640, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink", cv2.CAP_GSTREAMER)

def save_gps_data():
    global lat_values, lon_values, last_save_time
    
    # Obtener los valores de latitud y longitud desde el GPS
    lat = vehicle.location.global_frame.lat
    lon = vehicle.location.global_frame.lon
    
    # Agregar los valores de latitud y longitud a sus respectivas listas
    lat_values.append(lat)
    lon_values.append(lon)
    
    # Comprobar si ha pasado un segundo desde el último guardado
    current_time = time.time()
    if current_time - last_save_time >= 1:
        # Calcular el promedio de latitud y longitud durante este segundo
        avg_lat = sum(lat_values) / len(lat_values)
        avg_lon = sum(lon_values) / len(lon_values)
        
        # Obtener la marca de tiempo
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Guardar los promedios de latitud y longitud en el archivo
        with open("gps_datos.txt", "a") as file:
            file.write(f"{timestamp} - Latitud Promedio: {avg_lat:.6f}, Longitud Promedio: {avg_lon:.6f}\n")
            file.flush()  # Asegurarse de que se escriba inmediatamente
        
        # Limpiar las listas de valores para el siguiente segundo
        lat_values.clear()
        lon_values.clear()
        
        # Actualizar el tiempo de guardado
        last_save_time = current_time


yaw_values = deque()
last_save_time2 = time.time()

def save_compass_data():
    global yaw_values, last_save_time2
    
    yaw = vehicle.attitude.yaw
    yaw_grados = math.degrees(yaw)
    
    yaw_values.append(yaw_grados)
    
    current_time = time.time()
    if current_time - last_save_time2 >= 1:
        avg_yaw = sum(yaw_values) / len(yaw_values)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open("brujula_datos.txt", "a") as file:
            file.write(f"{timestamp} - Yaw Promedio: {avg_yaw:.2f} grados\n")
            file.flush()
        
        yaw_values.clear()
        last_save_time2 = current_time

def show_radar_visualization(distances):
    # Crear una ventana para la visualización estilo radar
    radar_size = 400
    radar_image = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)

    # Centro del radar
    center = (radar_size // 2, radar_size - 20)  # El centro está en la parte inferior
    max_radius = radar_size // 2
    max_distance = 250  # Valor máximo de distancia para los sensores (en cm)

    # Dibujar los límites del radar (semicírculo)
    cv2.ellipse(radar_image, center, (max_radius, max_radius), 0, 180, 360, (0, 255, 0), 1)

    # Ángulos de cada sensor según la disposición física real
    # Separamos ligeramente los sensores a 90 grados para reflejar la distancia entre ellos
    sensor_angles = [180, 135, 90, 90, 45, 0]  # Ángulos en grados para cada sensor

    # Ajuste para los sensores a 90 grados, separándolos un poco
    # La separación será en el eje X, por ejemplo, uno estará a 85° y el otro a 95°
    sensor_angles[3] = 85  # Primer sensor a 90 grados
    sensor_angles[2] = 95  # Segundo sensor a 90 grados

    # Dibujar el rover en el centro
    rover_width = 40
    rover_height = 30
    cv2.rectangle(radar_image, 
                  (center[0] - rover_width // 2, center[1] - rover_height // 2),
                  (center[0] + rover_width // 2, center[1] + rover_height // 2),
                  (255, 255, 255), 2)  # Rectángulo blanco representando el rover
    # Dibujar ruedas
    wheel_radius = 5
    cv2.circle(radar_image, (center[0] - rover_width // 4, center[1] + rover_height // 2), wheel_radius, (255, 255, 255), -1)  # rueda izquierda
    cv2.circle(radar_image, (center[0] + rover_width // 4, center[1] + rover_height // 2), wheel_radius, (255, 255, 255), -1)  # rueda derecha

    # Dibujar las líneas de los sensores y las distancias
    for i, distance in enumerate(distances):
        # Normaliza la distancia para ajustarla al radio del radar
        normalized_distance = int((distance / max_distance) * max_radius)
        if normalized_distance > max_radius:
            normalized_distance = max_radius

        # Calcular la posición (x, y) en la imagen del radar
        angle = math.radians(sensor_angles[i])
        x = int(center[0] + normalized_distance * math.cos(angle))
        y = int(center[1] - normalized_distance * math.sin(angle))

        # Calcular el color en función de la distancia (de verde a rojo)
        color = (0, 255 - int(normalized_distance / 2), int(normalized_distance / 2))  # Gradiente de color

        # Dibujar la línea desde el centro hasta la posición calculada
        cv2.line(radar_image, center, (x, y), color, 2)  # Línea con grosor constante
        cv2.circle(radar_image, (x, y), 5, color, -1)  # Marcador en el final de la línea

        # Mostrar la distancia en la línea
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(radar_image, f"{distance} cm", 
                    (x + 10, y - 10), 
                    font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)  # Distancia más legible

    # Mostrar la imagen del radar
    cv2.imshow('Radar Visualization', radar_image)
    cv2.waitKey(1)


# Crear ventana principal para visualización
cv2.namedWindow('Detección de fresas', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detección de fresas', 960, 360)  # Ventana combinada para las dos cámaras

print('Press \'q\' to stop')
stop_event = threading.Event()

try:
    while True:
        print("INFO: Conectando al vehículo.")
        timeout = time.time() + 30  # Tiempo límite para la conexión
        while not vehicle_connect():
            if time.time() > timeout:
                print("ERROR: No se pudo conectar con el vehículo.")
                break
        else:
            print("INFO: Vehículo conectado.")

        # Cambiar al modo HOLD antes de empezar
        change_vehicle_mode('AUTO')

        # Iniciar el hilo de sensores
        sensor_thread = threading.Thread(target=sensor_thread, args=(stop_event,))
        sensor_thread.daemon = True
        sensor_thread.start()

        # Captura y procesamiento de cámaras
        while True:
            ret1, frame1 = camera1.read()
            ret2, frame2 = camera2.read()

            if not ret1 or not ret2:
                print("ERROR: No se pudieron capturar frames de las cámaras.")
                break

            # Aplicar filtro LAB y obtener contornos
            contours_ripe_1, contours_unripe_1 = SBFilter(frame1)
            contours_ripe_2, contours_unripe_2 = SBFilter(frame2)

            # Convertir frames a CUDA
            try:
                img1 = jetson.utils.cudaFromNumpy(frame1.astype(np.float32))
                img2 = jetson.utils.cudaFromNumpy(frame2.astype(np.float32))
            except Exception as e:
                print(f"ERROR: Fallo al convertir frames a CUDA: {e}")
                break

            # Detección con la red
            detections1 = net.Detect(img1)
            detections1 = [d for d in detections1 if d.Confidence > confidence_threshold]
            detections1 = non_max_suppression(detections1)

            detections2 = net.Detect(img2)
            detections2 = [d for d in detections2 if d.Confidence > confidence_threshold]
            detections2 = non_max_suppression(detections2)

            # Dibujar detecciones
            def draw_detections(detections, frame):
                for detect in detections:
                    Id = detect.ClassID
                    item = net.GetClassDesc(Id) or "Desconocido"

                    if item == "Ripe":
                        color = (0, 255, 0)
                        save_compass_data()
                        save_gps_data()
                    elif item == "Unripe":
                        color = (0, 165, 255)
                    else:
                        color = (0, 0, 255)

                    top_left = (int(detect.Left), int(detect.Top))
                    bottom_right = (int(detect.Right), int(detect.Bottom))

                    cv2.rectangle(frame, top_left, bottom_right, color, 3)
                    confidence_text = f"{int(detect.Confidence * 100)}%"
                    cv2.putText(frame, f"{item} ({confidence_text})", (int(detect.Left), int(detect.Top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            output_frame1 = frame1.copy()
            output_frame2 = frame2.copy()

            draw_detections(detections1, output_frame1)
            draw_detections(detections2, output_frame2)

            # Concatenar frames
            if output_frame1.shape[:2] == output_frame2.shape[:2]:
                combined_frame = np.hstack((output_frame1, output_frame2))
            else:
                print("ERROR: Las dimensiones de los frames no coinciden.")
                break

            # Mostrar resultados
            cv2.imshow('Detección de fresas', combined_frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()  # Finalizar hilo
                break

finally:
    stop_event.set()
    sensor_thread.join()
    camera1.release()
    camera2.release()
    cv2.destroyAllWindows()
    print("INFO: Recursos liberados correctamente.")
