import time
from dronekit import connect, VehicleMode
from pymavlink import mavutil
import serial
import threading

# Conexión del vehículo
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

    # Evaluar los sensores de la izquierda y derecha
    left_sensors = distances[0:3]  # Sensores 1, 2, 3
    right_sensors = distances[3:6]  # Sensores 4, 5, 6

    # Detectar el mínimo en ambos lados
    min_left = min(left_sensors)
    min_right = min(right_sensors)

    # Si detecta obstáculo, elige el lado con menor distancia
    if min_left < 20 or min_right < 20:
        change_vehicle_mode('MANUAL')
        
        if min_left < min_right:
            print("Obstáculo detectado a la izquierda, maniobrando a la derecha.")
            send_rc_control(2000, 2000)  # Aumentar ligeramente la velocidad hacia adelante y girar suavemente a la derecha
        else:
            print("Obstáculo detectado a la derecha, maniobrando a la izquierda.")
            send_rc_control(2000, 900)  # Aumentar ligeramente la velocidad hacia adelante y girar suavemente a la izquierda

    else:
        change_vehicle_mode('HOLD')
        send_rc_control(1500, 1500)  # Detener el movimiento y enderezar dirección

# Función que maneja la ejecución concurrente
def sensor_thread():
    while True:
        maneuver_based_on_sensors()
        time.sleep(0.001)  # Reducir el tiempo de espera para una respuesta más rápida

# Conectar al vehículo
print("INFO: Conectando al vehículo.")
while not vehicle_connect():
    pass
print("INFO: Vehículo conectado.")

# Cambiar al modo HOLD antes de empezar
change_vehicle_mode('HOLD')

# Ejecutar la lectura de sensores y maniobras en un hilo separado
sensor_thread = threading.Thread(target=sensor_thread)
sensor_thread.daemon = True  # Esto hace que el hilo termine cuando el principal termine
sensor_thread.start()

# Aquí podemos ejecutar otras tareas si es necesario o simplemente dejarlo correr
while True:
    time.sleep(0.001)  # El hilo de los sensores sigue funcionando en segundo plano

