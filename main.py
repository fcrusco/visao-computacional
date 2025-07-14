import cv2
from ultralytics import YOLO

# Carrega o modelo YOLOv8 (pode ser yolov8n.pt, yolov8s.pt, yolov8m.pt, etc)
model = YOLO("yolov8n.pt")  # Baixa o modelo automaticamente na primeira vez

# Abre a câmera padrão (0 = webcam integrada)
cap = cv2.VideoCapture(0)

# Loop principal
while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a câmera.")
        break

    # Detecta objetos no frame
    results = model(frame, stream=True)

    # Itera sobre os resultados
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Coordenadas da caixa delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Desenha o retângulo e rótulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f'{label} ({conf:.2f})'
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostra o frame com as marcações
    cv2.imshow("Detecção de Objetos (YOLOv8)", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
