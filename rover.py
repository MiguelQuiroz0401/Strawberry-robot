#!/usr/bin/env python3

import sys
import argparse
import time
import threading
import math
from dronekit import connect, VehicleMode
from pymavlink import mavutil
from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

# Función para conectar al Pixhawk
def connectMyCopter(connection_string):
    vehicle = connect(connection_string, baud=57600, wait_ready=True)
    return vehicle

# Función para cambiar el modo de vuelo
def set_mode(vehicle, mode_name):
    vehicle.mode = VehicleMode(mode_name)
    while vehicle.mode.name != mode_name:
        print(f"Waiting for vehicle to enter {mode_name} mode")
        time.sleep(1)
    print(f"Vehicle now in {mode_name} mode")

# Función para enviar señal PWM al canal RC1
def send_pwm_to_rc1(vehicle, pwm_value):
    vehicle.channels.overrides['1'] = pwm_value
    print(f"PWM {pwm_value} enviado al canal RC1")

# Función para controlar el servo en un hilo separado
def control_servo(vehicle, pwm_value):
    set_mode(vehicle, "MANUAL")
    send_pwm_to_rc1(vehicle, pwm_value)
    time.sleep(1)  # Mantener el servo en esta posición por 1 segundo
    send_pwm_to_rc1(vehicle, 1500)  # Restablecer a posición neutra
    set_mode(vehicle, "GUIDED")  # Volver al modo GUIDED

# Función para detectar fresas y controlar el servo
def detect_and_control(vehicle, net, input, output):
    while True:
        img = input.Capture()
        if img is None:
            continue

        detections = net.Detect(img, overlay="box,labels,conf")
        print(f"Detected {len(detections)} objects in image")

        for detection in detections:
            if detection.ClassID == 1:  # Suponiendo que la clase 1 es fresa madura
                print("Fresa madura detectada")
                # Ejecutar el control del servo en un hilo separado
                threading.Thread(target=control_servo, args=(vehicle, 2000)).start()
                break

        output.Render(img)
        output.SetStatus(f"{args.model} | Network {net.GetNetworkFPS()} FPS")
        net.PrintProfilerTimes()

        if not input.IsStreaming() or not output.IsStreaming():
            break

# Parseo de argumentos
parser = argparse.ArgumentParser(description="Controlar un rover basado en la detección de fresas.",
                                 formatter_class=argparse.RawTextHelpFormatter,
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())
parser.add_argument("input", type=str, default="", nargs='?', help="URI de la fuente de video")
parser.add_argument("output", type=str, default="", nargs='?', help="URI del video de salida")
parser.add_argument("--model", type=str, default="models/fresas/ssd-mobilenet.onnx", help="Ruta al modelo ONNX")
parser.add_argument("--labels", type=str, default="models/fresas/labels.txt", help="Ruta al archivo de etiquetas")
parser.add_argument("--input-blob", type=str, default="input_0", help="Nombre del blob de entrada en el modelo ONNX")
parser.add_argument("--output-cvg", type=str, default="scores", help="Nombre del blob de cobertura de salida")
parser.add_argument("--output-bbox", type=str, default="boxes", help="Nombre del blob de cajas de salida")
parser.add_argument("--threshold", type=float, default=0.5, help="Umbral de detección mínimo")
parser.add_argument("--connect", type=str, required=True, help="Cadena de conexión al Pixhawk")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# Crear fuentes de video y salida
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# Cargar la red de detección de objetos
net = detectNet(model=args.model, labels=args.labels, input_blob=args.input_blob,
                output_cvg=args.output_cvg, output_bbox=args.output_bbox, threshold=args.threshold)

# Conectar al Pixhawk
vehicle = connectMyCopter(args.connect)

# Iniciar el control del rover
detect_and_control(vehicle, net, input, output)

