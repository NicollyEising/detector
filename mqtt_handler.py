import paho.mqtt.client as mqtt

class MqttHandler:
    def __init__(self, config, video_processor, data_queue):
        try:
            self.BLYNK_AUTH_TOKEN = config["blynk_auth_token"]
            self.BROKER = config["broker"]
            self.PORT = config["port"]
            self.client = mqtt.Client()
            self.client.username_pw_set(config["mqtt_username"], self.BLYNK_AUTH_TOKEN)
            self.client.on_connect = self.on_connect
            self.client.tls_set()
            self.client.connect(self.BROKER, self.PORT, 60)
            self.video_processor = video_processor
            self.data_queue = data_queue
        except Exception as e:
            print(f"Erro na inicialização do MQTT: {e}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Conectado ao Blynk com sucesso!")
        else:
            print(f"Falha na conexão com o código {rc}")

    def send_to_blynk(self):
        try:
            count_left = self.video_processor.count_left
            count_right = self.video_processor.count_right
            self.client.publish("ds/esquerda", str(count_left))
            self.client.publish("ds/direita", str(count_right))
        except Exception as e:
            print(f"Erro ao enviar dados para o Blynk: {e}")

    def process_mqtt(self):
        while True:
            try:
                if not self.data_queue.empty():
                    # Obtém dados da fila
                    count_left, count_right = self.data_queue.get()

                    # Envia os dados para o Blynk
                    self.send_to_blynk()

                self.client.loop()
            except Exception as e:
                print(f"Erro no processamento MQTT: {e}")
                break  # Finaliza o loop em caso de erro
