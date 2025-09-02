from dronekit import connect, VehicleMode
import serial
import time
import re  # Importar la librería para expresiones regulares

# Conectar al Pixhawk
print("Conectando al Pixhawk en /dev/ttyTHS1...")
vehicle = connect('/dev/ttyTHS1', baud=57600, wait_ready=True, timeout=60)
print("Pixhawk conectado.")

# Conectar al Arduino
print("Conectando al Arduino en /dev/ttyACM0...")
arduino = serial.Serial('/dev/ttyACM0', 9600)
time.sleep(2)
print("Arduino conectado.")

# Cambiar al modo MANUAL y esperar la confirmación
print("Cambiando el Pixhawk a modo MANUAL...")
vehicle.mode = VehicleMode("MANUAL")
while not vehicle.mode.name == 'MANUAL':
    print("Esperando a cambiar a modo MANUAL...")
    time.sleep(1)
print("Vehículo ahora en modo MANUAL.")

# Canal de dirección y valores PWM para el servo
steering_channel = 3
neutral_pwm = 1500  # Posición neutral
turn_left_pwm = 2000  # Giro a la izquierda para el Sensor 1
turn_right_pwm = 850  # Giro a la derecha para el Sensor 2
distance_threshold = 20  # Nueva distancia mínima en cm

try:
    while True:
        if arduino.in_waiting > 0:
            # Leer y procesar datos del Arduino
            distance_data = arduino.readline().decode().strip()
            
            # Buscar todos los valores de distancia en la cadena
            distances = re.findall(r'Distancia \d+: (\d+) cm', distance_data)
            
            if distances:
                # Convertir las distancias a números enteros
                distances = [int(d) for d in distances]
                print(f"Distancias leídas: {distances} cm")

                # Control de giro según el sensor (verificamos que haya al menos 3 lecturas)
                if len(distances) >= 1 and distances[0] < distance_threshold:
                    # Sensor 1 detecta un obstáculo cercano
                    print("Obstáculo detectado en Sensor 1: Girando a la izquierda (PWM 2000)")
                    vehicle.channels.overrides[steering_channel] = turn_left_pwm

                elif len(distances) >= 2 and distances[1] < distance_threshold:
                    # Sensor 2 detecta un obstáculo cercano
                    print("Obstáculo detectado en Sensor 2: Girando a la derecha (PWM 850)")
                    vehicle.channels.overrides[steering_channel] = turn_right_pwm

                else:
                    # Si no hay obstáculos en Sensor 1 o 2, mantener la dirección neutral
                    print("Sin obstáculos en sensores 1 y 2: Manteniendo dirección recta (PWM 1500)")
                    vehicle.channels.overrides[steering_channel] = neutral_pwm

                # Advertencia para el Sensor 3
                if len(distances) >= 3 and distances[2] < distance_threshold:
                    print("Advertencia: Objeto muy próximo detectado en Sensor 3.")

            else:
                print(f"Formato inesperado de datos: {distance_data}")

            # Breve pausa antes de la siguiente lectura
            time.sleep(0.1)

except KeyboardInterrupt:
    print("Interrumpido por el usuario. Restaurando servo a posición neutral.")
    vehicle.channels.overrides[steering_channel] = neutral_pwm

finally:
    # Limpiar y cerrar conexiones
    arduino.close()
    vehicle.close()
    print("Conexiones cerradas.")

