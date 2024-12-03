"""Módulo para processamento de vídeo e contagem de pessoas usando YOLO."""

import cv2
import numpy as np

class VideoProcessor:
    """Classe responsável pelo processamento de vídeo e detecção de pessoas."""

    def __init__(self, config, data_queue):
        """Inicializa o processador de vídeo com os parâmetros de configuração fornecidos."""
        self.cap = cv2.VideoCapture(config["video_source"])
        self.net = cv2.dnn.readNet(config["yolo_weights"], config["yolo_cfg"])
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [
            self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()
        ]
        self.classes = self._load_classes(config["names_path"])

        # Contadores e estados
        self.count_left = 0
        self.count_right = 0
        self.count_left_updated = False
        self.count_right_updated = False
        self.crossed_people = {}
        self.line_position = None
        self.people = {}
        self.data_queue = data_queue

        # Configuração inicial
        ret, frame = self.cap.read()
        if ret:
            _, width, _ = frame.shape
            self.line_position = width // 2

        self.fps = 20

    def _load_classes(self, names_path):
        """Carrega as classes do arquivo names_path."""
        with open(names_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]

    def detect_objects(self, frame):
        """Realiza a detecção de objetos em um frame usando YOLO."""
        height, width, _ = frame.shape
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)
        return outputs, width, height

    def apply_nms(
        self, outputs, width, height, confidence_threshold=0.5, nms_threshold=0.4
    ):
        """Aplica Non-Maximum Suppression para filtrar detecções."""
        boxes, confidences, class_ids = [], [], []
        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    center_x, center_y = (
                        int(detection[0] * width),
                        int(detection[1] * height),
                    )
                    w, h = int(detection[2] * width), int(detection[3] * height)
                    x1, y1 = center_x - w // 2, center_y - h // 2
                    x2, y2 = center_x + w // 2, center_y + h // 2
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, confidence_threshold, nms_threshold
        )
        return boxes, confidences, class_ids, indices.flatten() if indices is not None else []

    def count_people(self, frame, outputs, width, height):
        """Conta as pessoas que cruzam a linha definida."""
        boxes, confidences, class_ids, indices = self.apply_nms(
            outputs, width, height
        )
        new_people = {}

        for i in indices:
            if class_ids[i] == 0:  # Classe 0 é pessoa
                x1, y1, x2, y2 = boxes[i]
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    'Person',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2,
                )
                new_people[center_x] = (center_x, center_y, x1, y1, x2, y2)

        self.update_people(new_people)

    def update_people(self, new_people):
        """Atualiza a lista de pessoas com base nas novas detecções."""
        updated_people = {}
        for center_x, new_person in new_people.items():
            matched = False
            for old_person_id, person in self.people.items():
                distance = np.sqrt(
                    (person[0] - new_person[0]) ** 2 + (person[1] - new_person[1]) ** 2
                )
                if distance < 100:
                    updated_people[old_person_id] = new_person
                    matched = True
                    break
            if not matched:
                updated_people[len(updated_people)] = new_person
        self.people = updated_people

    def process_video(self):
        """Processa o vídeo e conta o número de pessoas cruzando a linha."""
        previous_positions = {}
        frame_count = 0

        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    raise StopIteration("Processamento de vídeo concluído.")

                frame_count += 1
                if frame_count % self.fps == 0:
                    outputs, width, height = self.detect_objects(frame)
                    self.count_people(frame, outputs, width, height)

                    for person_id, person in self.people.items():
                        person_center_x = person[0]
                        if person_id in previous_positions:
                            if previous_positions[person_id] < self.line_position <= person_center_x:
                                self.count_right += 1
                                self.count_right_updated = True
                                self.crossed_people[person_id] = 'right'
                            elif previous_positions[person_id] > self.line_position >= person_center_x:
                                self.count_left += 1
                                self.count_left_updated = True
                                self.crossed_people[person_id] = 'left'

                        previous_positions[person_id] = person_center_x

                    if self.count_left_updated or self.count_right_updated:
                        self.data_queue.put((self.count_left, self.count_right))
                        self.count_left_updated = False
                        self.count_right_updated = False

                    cv2.line(
                        frame,
                        (self.line_position, 0),
                        (self.line_position, height),
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f'Left: {self.count_left}',
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f'Right: {self.count_right}',
                        (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow('Object Detection - Counting People', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except StopIteration as e:
                print(e)
                break
            except cv2.error as e:
                print(f"Erro do OpenCV: {e}")
                break
            except Exception as e:
                print(f"Erro no processamento: {e}")
                break

        self.cap.release()
        cv2.destroyAllWindows()
