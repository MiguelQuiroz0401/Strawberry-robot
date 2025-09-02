from dronekit import connect
import time
from datetime import datetime
import math

# Conectar al Pixhawk
print("Conectando al Pixhawk en /dev/ttyTHS1...")
vehicle = connect('/dev/ttyTHS1', baud=57600, wait_ready=True, timeout=60)
print("Pixhawk conectado.")

# Abrir el archivo en modo de escritura (append) para no sobrescribir los datos previos
with open("brujula_datos.txt", "a") as file:
    try:
        while True:
            # Obtener los valores de la brújula (yaw) en radianes
            yaw = vehicle.attitude.yaw
            yaw_grados = math.degrees(yaw)  # Convertir de radianes a grados
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Escribir en el archivo en grados
            file.write(f"{timestamp} - Yaw: {yaw_grados} grados\n")
            file.flush()  # Asegurarse de que se escriba inmediatamente


            print(f"{timestamp} - Yaw: {yaw_grados} grados") # aaaaa



            # Pausa antes de la próxima lectura
            time.sleep(0.02)

    except KeyboardInterrupt:
        print("Interrumpido por el usuario.")

    finally:
        # Cerrar la conexión al Pixhawk al finalizar
        vehicle.close()
        print("Conexión al Pixhawk cerrada.")

