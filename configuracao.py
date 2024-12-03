
from ultralytics import YOLO

# Carregar um modelo pré-treinado YOLOv8 (se disponível)
model = YOLO("yolov8n.pt")  # Você pode trocar para outro modelo se precisar (yolov8s, yolov8m, etc.)

# Definir o diretório de dados para treinamento (se você tiver seus próprios dados)
# Aqui, assumimos que você tem um conjunto de dados já preparado em formato YOLO (com imagens e labels)
# Se for treinar do zero, você precisa organizar seus dados em 'train' e 'val' folders.

# Treinamento com seus próprios dados
model.train(data="data.yaml", epochs=50, imgsz=640)

# Ou, se não for treinar e usar um modelo já treinado para inferência:
# model = YOLO("yolov8n.pt")  # Carregar um modelo pré-treinado

# Realizar inferência em uma imagem
results = model.predict("image.jpg")  # Substitua com o caminho para a sua imagem
results.show()  # Exibe a imagem com a detecção

# Salvar o modelo após o treinamento
model.save("model_yolov8.pt")
