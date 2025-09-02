import cv2
import subprocess
import numpy as np
import os

def run_detectnet(image_path):
    # Define the command
    command = [
        "detectnet",
        "--model=models/fresas/ssd-mobilenet.onnx",
        "--labels=models/fresas/labels.txt",
        "--input-blob=input_0",
        "--output-cvg=scores",
        "--output-bbox=boxes",
        image_path
    ]
    
    # Run the command and capture the output
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"Error: {stderr}")
        return None
    
    return stdout

def parse_detectnet_output(output):
    # Parse the detectnet output to extract bounding boxes and labels
    # For demonstration, we'll just return dummy data
    # You should replace this with actual parsing logic
    boxes = []
    labels = []
    # Example output parsing
    for line in output.splitlines():
        if line.startswith('box'):
            parts = line.split()
            boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            labels.append(parts[5])
    return boxes, labels

def draw_boxes(frame, boxes, labels):
    # Draw bounding boxes on the frame
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)  # 0 is the default camera
    
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Save the frame to a temporary file
        temp_file_path = "/tmp/temp_image.jpg"
        cv2.imwrite(temp_file_path, frame)
        
        # Run detectnet and get the result
        output = run_detectnet(temp_file_path)
        if output:
            boxes, labels = parse_detectnet_output(output)
            frame = draw_boxes(frame, boxes, labels)
        
        # Display the frame
        cv2.imshow('Frame', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

