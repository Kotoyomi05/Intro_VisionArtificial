from ultralytics import YOLO

# Modelo pequeño recomendado para empezar (Nano).
# La primera vez se descargará automáticamente de internet.
modelo = YOLO("yolov8n.pt")

# source=0 usa la webcam principal.
# show=True muestra la ventana con las detecciones en vivo.
# conf=0.5 filtra detecciones con confianza menor al 50%.
modelo.predict(source=0, show=True, conf=0.5)