import freenect
import cv2
import numpy as np
from dronekit import connect, VehicleMode
import time
import threading

# Configuración de parámetros
DISTANCE_THRESHOLD = 70  # Distancia mínima para la detección
STEERING_LEFT = 850      # PWM para girar a la izquierda
STEERING_RIGHT = 2000    # PWM para girar a la derecha
STEERING_NEUTRAL = 1500  # PWM para posición neutral

# Captura de Profundidad
def pretty_depth(depth):
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    return depth.astype(np.uint8)

def get_depth_frame():
    depth, _ = freenect.sync_get_depth()
    if depth is not None:
        return pretty_depth(depth)
    return None

# Control del Rover
def connectMyCopter(connection_string):
    print("Conectando al Pixhawk...")
    vehicle = connect(connection_string, baud=57600, wait_ready=True, timeout=60)
    print("Conexión exitosa.")
    return vehicle

def set_mode(vehicle, mode_name):
    vehicle.mode = VehicleMode(mode_name)
    while vehicle.mode.name != mode_name:
        print(f"Esperando para entrar en modo {mode_name}...")
        time.sleep(1)
    print(f"Vehículo ahora en modo {mode_name}.")

def send_pwm_to_rc3(vehicle, pwm_value):
    vehicle.channels.overrides['3'] = pwm_value
    print(f"PWM {pwm_value} enviado al canal RC3")

# Filtrado de Distancias
def smooth_distance(distance, history, num_samples=5):
    history.append(distance)
    if len(history) > num_samples:
        history.pop(0)
    return sum(history) / len(history)

# Función para visualizar y controlar el rover
def display_depth(vehicle):
    cv2.namedWindow('Depth Zones', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Depth Zones', 800, 600)

    left_distance_history = []
    right_distance_history = []

    while True:
        depth_frame = get_depth_frame()

        if depth_frame is not None:
            height, width = depth_frame.shape
            left_zone = depth_frame[:, :width // 2]
            right_zone = depth_frame[:, width // 2:]

            # Obtener las distancias mínimas en cada zona
            left_distance = np.min(left_zone)
            right_distance = np.min(right_zone)

            # Convertir los valores a distancia real
            left_distance_real = left_distance * 0.5  # Ajustar este factor según tu calibración
            right_distance_real = right_distance * 0.5

            # Suavizar las distancias
            left_distance_smooth = smooth_distance(left_distance_real, left_distance_history)
            right_distance_smooth = smooth_distance(right_distance_real, right_distance_history)

            print(f"Distancia izquierda (suavizada): {left_distance_smooth:.2f} cm, Distancia derecha (suavizada): {right_distance_smooth:.2f} cm")

            # Lógica de esquivación
            if left_distance_smooth < DISTANCE_THRESHOLD or right_distance_smooth < DISTANCE_THRESHOLD:
                set_mode(vehicle, "ACRO")  # Cambiar a modo ACRO
                if left_distance_smooth < right_distance_smooth:
                    print("Obstáculo a la izquierda, girando a la derecha.")
                    send_pwm_to_rc3(vehicle, STEERING_RIGHT)  # Girar a la derecha
                else:
                    print("Obstáculo a la derecha, girando a la izquierda.")
                    send_pwm_to_rc3(vehicle, STEERING_LEFT)  # Girar a la izquierda
            else:
                set_mode(vehicle, "MANUAL")  # Cambiar a modo MANUAL
                send_pwm_to_rc3(vehicle, STEERING_NEUTRAL)  # Estado neutral

            # Combinar y mostrar las zonas
            combined_frame = np.hstack((left_zone, right_zone))
            cv2.imshow('Depth Zones', combined_frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC para salir
            break

    cv2.destroyAllWindows()

def main():
    connection_string = "/dev/ttyTHS1"  # Ajustar esto
    vehicle = connectMyCopter(connection_string)

    if vehicle is None:
        print("No se pudo establecer la conexión con el vehículo.")
        return

    set_mode(vehicle, "ACRO")  # Iniciar en modo ACRO
    send_pwm_to_rc3(vehicle, STEERING_NEUTRAL)  # Estado neutral

    # Crear un hilo para la visualización y control
    display_thread = threading.Thread(target=display_depth, args=(vehicle,))
    display_thread.start()

    display_thread.join()  # Esperar a que el hilo termine

    # Cerrar conexión
    vehicle.close()
    freenect.sync_stop()

if __name__ == "__main__":
    main()

