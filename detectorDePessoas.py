import queue
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time
import threading

# Caminhos para os arquivos YOLO
yolo_cfg = 'C:/javaConteudo/DetectorDePessoas/n3_iot/pytorch-YOLOv4/cfg/yolov4.cfg'
yolo_weights = 'C:/javaConteudo/DetectorDePessoas/n3_iot/pytorch-YOLOv4/cfg/yolov4.weights'
names_path = 'C:/javaConteudo/DetectorDePessoas/n3_iot/coco.names'

# Carregar a rede YOLO
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Carregar os nomes das classes
with open(names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Inicializar a captura de vídeo
cap = cv2.VideoCapture("video.mp4")

# Variáveis de contador
count_left = 0
count_right = 0
people = []  # Lista de pessoas detectadas com coordenadas
person_ids = []  # Lista para armazenar IDs únicos das pessoas
crossed_people = {}  # Dicionário para armazenar IDs de pessoas que cruzaram a linha e direção

# Definir a linha central
ret, frame = cap.read()
if ret:
    height, width, _ = frame.shape
    line_position = width // 2  # Calcula o centro da resolução

# Dados do Blynk e MQTT
BLYNK_AUTH_TOKEN = 'N8jEZTMvcR_bwGyrEvkh188DdHyeimZQ'
BROKER = 'blynk.cloud'
PORT = 8883

# Flags para detectar alterações nos contadores
count_left_updated = False
count_right_updated = False

# Função de conexão MQTT
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Conectado ao Blynk com sucesso!")
    else:
        print(f"Falha na conexão com o código {rc}")

# Função de envio de contagens para o Blynk
def send_to_blynk(client, count_left, count_right):
    print(f"Enviando contagem para Blynk: Left={count_left}, Right={count_right}")
    client.publish("ds/esquerda", str(count_left))  # Contagem da esquerda
    client.publish("ds/direita", str(count_right))  # Contagem da direita

# Função para detectar objetos
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)
    return outputs, width, height

# Função para aplicar NMS
def apply_nms(outputs, width, height, confidence_threshold=0.5, nms_threshold=0.4):
    boxes = []
    confidences = []
    class_ids = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x1 = int(center_x - w / 2)
                y1 = int(center_y - h / 2)
                x2 = int(center_x + w / 2)
                y2 = int(center_y + h / 2)
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    if indices is not None and len(indices) > 0:
        return boxes, confidences, class_ids, indices.flatten()
    else:
        return boxes, confidences, class_ids, []

# Função para contar e desenhar objetos detectados
def count_people(frame, outputs, width, height):
    global count_left, count_right, count_left_updated, count_right_updated, people, person_ids, crossed_people
    new_people = []
    boxes, confidences, class_ids, indices = apply_nms(outputs, width, height)

    for i in indices if indices is not None else []:
        class_id = class_ids[i]
        if class_id == 0:  # Apenas pessoas
            x1, y1, x2, y2 = boxes[i]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(classes[class_id]), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            new_people.append((center_x, center_y, x1, y1, x2, y2))

    updated_people = []
    updated_person_ids = []
    for new_person in new_people:
        matched = False
        for i, person in enumerate(people):
            distance = np.sqrt((person[0] - new_person[0])**2 + (person[1] - new_person[1])**2)
            if distance < 100:
                updated_people.append((new_person[0], new_person[1], person[2], person[3], person[4], person[5]))
                updated_person_ids.append(person_ids[i])
                matched = True
                break
        if not matched:
            updated_people.append(new_person)
            updated_person_ids.append(len(updated_person_ids))
    people[:] = updated_people
    person_ids[:] = updated_person_ids

# Função para processar vídeo
def video_thread():
    global count_left, count_right, count_left_updated, count_right_updated
    previous_positions = {}  # Dicionário para armazenar as posições anteriores de cada pessoa

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        outputs, width, height = detect_objects(frame)
        count_people(frame, outputs, width, height)

        # Contagem de pessoas cruzando a linha
        for i, person in enumerate(people):
            person_center_x = person[0]
            person_id = person_ids[i]

            # Verifica se a pessoa já estava registrada (em um frame anterior)
            if person_id in previous_positions:
                # Verifica se houve cruzamento da linha
                if previous_positions[person_id] < line_position and person_center_x >= line_position:
                    # Cruzou da esquerda para a direita
                    print(f"Pessoa {person_id} cruzou para a direita")
                    count_right += 1
                    crossed_people[person_id] = 'right'
                    count_right_updated = True  # Marcar que o contador da direita foi atualizado
                elif previous_positions[person_id] > line_position and person_center_x <= line_position:
                    # Cruzou da direita para a esquerda
                    print(f"Pessoa {person_id} cruzou para a esquerda")
                    count_left += 1
                    crossed_people[person_id] = 'left'
                    count_left_updated = True  # Marcar que o contador da esquerda foi atualizado

            # Atualizar a posição anterior
            previous_positions[person_id] = person_center_x

        # Atualizar pessoas que saíram da área
        for person_id in list(crossed_people.keys()):
            person_in_frame = any(person_id == pid for pid in person_ids)
            if not person_in_frame:
                del crossed_people[person_id]
                del previous_positions[person_id]  # Limpar a posição anterior também

        # Desenhar a linha central
        cv2.line(frame, (line_position, 0), (line_position, height), (0, 0, 255), 2)

        # Exibir o número de pessoas à esquerda e à direita
        cv2.putText(frame, f'Left: {count_left}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f'Right: {count_right}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Mostrar a imagem com detecção e contagem
        cv2.imshow('Object Detection - Counting People', frame)

        # Interromper o loop quando pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Função para processar MQTT
def mqtt_thread():
    global count_left_updated, count_right_updated
    client = mqtt.Client()
    client.username_pw_set("device", BLYNK_AUTH_TOKEN)
    client.on_connect = on_connect
    client.tls_set()
    client.connect(BROKER, PORT, 60)
    while True:
        if count_left_updated or count_right_updated:
            print(f"Enviando para o Blynk: Left={count_left}, Right={count_right}")
            send_to_blynk(client, count_left, count_right)
            count_left_updated = False  # Resetar flag após envio
            count_right_updated = False  # Resetar flag após envio
        client.loop()
        time.sleep(1)

# Criar e iniciar threads
video_thread_instance = threading.Thread(target=video_thread)
mqtt_thread_instance = threading.Thread(target=mqtt_thread)

video_thread_instance.start()
mqtt_thread_instance.start()

video_thread_instance.join()
mqtt_thread_instance.join()

cap.release()
cv2.destroyAllWindows()
