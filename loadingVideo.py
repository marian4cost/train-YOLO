#importando bibliotecas
import os
from ultralytics import YOLO
import cv2

#diretório onde a imagem está armazenada
IMAGES_DIR = os.path.join('.', 'image')
image_path = os.path.join(IMAGES_DIR, '001.jpg')
image_path_out = '{}out.jpg'.format(image_path)

#abertura/leitura da imagem, obtendo as dimensões (altura e largura); logo após cria o objeto para a saída do resultado
cap = cv2.VideoCapture(image_path) 
ret, frame = cap.read() 
H, W, _ = frame.shape
out = cv2.VideoWriter(image_path_out, cv2.VideoWriter_fourcc(*'JPGV'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))

#caminho para o modelo pré-treinado do YOLO
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path) 

#limite de confiança para a filtragem de detecções (só irá afirmar se o valor for maior que o definido)
threshold = 0.1

#faz a leitura da imagem pixel por pixel,passa para o modelo YOLO para a detecção e em casos de identificação desenha uma caixa delimitadora e rótulos no objeto com base no que foi rótulado
while ret:
    results = model(frame)[0]
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

#libera os recursos de vídeo/imagem e fecha todas as janelas abertas/usadas pelo openCV
cap.release()
out.release()
cv2.destroyAllWindows()