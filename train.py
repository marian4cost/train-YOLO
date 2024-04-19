from ultralytics import YOLO #importando o YOLO

model = YOLO("yolov8n.yaml") #iniciando um objeto YOLO com base no arquivo .yaml

results = model.train(data="config.yaml", epochs=1) #iniciando o treinamento do YOLO com base no arquivo .yaml