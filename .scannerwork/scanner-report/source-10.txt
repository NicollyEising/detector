"""Script principal para detecção de pessoas e envio de dados via MQTT."""

import json
import queue
import concurrent.futures
from yolo_detector import VideoProcessor
from mqtt_handler import MqttHandler

def main():
    """Função principal que inicializa e coordena o processamento de vídeo e MQTT."""
    try:
        print("Iniciando o script...")

        # Carregar configurações do arquivo JSON
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        # Criar fila para comunicação entre as threads
        data_queue = queue.Queue()

        # Inicializar as classes
        video_processor = VideoProcessor(config, data_queue)
        mqtt_handler = MqttHandler(config, video_processor, data_queue)

        # Usar ThreadPoolExecutor para executar as funções de forma assíncrona
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Enviar o processamento do vídeo para uma thread
            video_future = executor.submit(video_processor.process_video)
            
            # Enviar o processo de MQTT para outra thread
            mqtt_future = executor.submit(mqtt_handler.process_mqtt)

            # Esperar até que ambas as tarefas sejam concluídas
            concurrent.futures.wait([video_future, mqtt_future])

    except FileNotFoundError as e:
        print(f"Erro ao abrir arquivo: {e}")
    except ValueError as e:
        print(f"Erro de valor: {e}")
    except Exception as e:
        print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    main()
