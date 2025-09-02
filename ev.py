import os
import subprocess

# Configura aquí las rutas y parámetros
model_dir = "/home/miguel/backup/jetson-inference/python/training/detection/ssd/models/fresas"
dataset = "/home/miguel/backup/jetson-inference/python/training/detection/ssd/data/fresas"
eval_dir = "/home/miguel/backup/jetson-inference/python/training/detection/ssd/eval_results"
net_type = "mb1-ssd"

# Filtra todos los archivos .pth en la carpeta
model_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pth"))

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    print(f"Evaluando modelo: {model_file}")
    
    command = [
        "python3", "eval_ssd.py",
        "--net", net_type,
        "--model", model_path,
        "--dataset_type", "voc",
        "--dataset", dataset,
        "--iou_threshold", "0.5",
        "--nms_method", "hard",
        "--eval_dir", eval_dir,
        "--use_cuda", "True"
    ]
    
    subprocess.run(command)
