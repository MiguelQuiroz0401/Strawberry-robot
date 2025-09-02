import cv2
import numpy as np

# Cargar la imagen en formato RGB
img_rgb = cv2.imread('/home/miguel/backup/jetson-inference/python/training/detection/ssd/data/fresas/JPEGImages/1_png.rf.7574b656bd9f16ab1d280230b159bc4c.jpg')

# Convertir la imagen de RGB a HSV
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)

# Mostrar la imagen en HSV (opcional)
cv2.imshow('HSV Image', img_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

