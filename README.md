# Sistema de Contagem de Passageiros de Ônibus com YOLO e MQTT

Este projeto tem como objetivo contar o número de passageiros que entram e saem de um ônibus, utilizando a tecnologia YOLO para detecção de pessoas, OpenCV para processamento de vídeo, e MQTT para envio de dados para uma plataforma IoT como o Blynk.

## Funcionalidades

- **Detecção de Pessoas**: Utiliza YOLOv4 para detectar passageiros em tempo real a partir de um vídeo do interior do ônibus.
- **Contagem de Passageiros**: Conta o número de passageiros que entram e saem do ônibus, baseando-se na detecção de pessoas cruzando uma linha imaginária.
- **Envio de Dados via MQTT**: Envia as contagens de passageiros para um broker MQTT (como o Blynk IoT), permitindo monitoramento remoto.

## Estrutura do Projeto

O projeto é composto por três módulos principais:

- **main.py**: Script principal que inicializa e coordena o processamento de vídeo e o envio de dados via MQTT.
- **yolo_detector.py**: Módulo responsável pela detecção de objetos (passageiros) no vídeo utilizando YOLO.
- **mqtt_handler.py**: Módulo que gerencia a conexão MQTT e o envio de dados para a plataforma IoT.

## Pré-requisitos

Antes de executar o projeto, é necessário ter o seguinte instalado:

- Python 3.x

Para instalar as dependências, execute o comando abaixo:

```pip install -r requirements.txt```

# Arquivo `requirements.txt`

O arquivo `requirements.txt` contém todas as bibliotecas necessárias para o funcionamento do projeto:

- opencv-python
- paho-mqtt
- numpy

# YOLOv4

Certifique-se de ter os arquivos de configuração (`.cfg`), pesos (`.weights`) e o arquivo de classes (`.names`) do modelo YOLOv4. Você pode baixar esses arquivos do site oficial do YOLO.

# Como Executar

Para iniciar o sistema de contagem de passageiros, basta executar o script principal:

```python main.py```

Isso iniciará o processamento de vídeo e a detecção de passageiros, além de enviar os dados de contagem via MQTT.

# Descrição do Código

- **main.py**:
  - Carrega as configurações a partir de um arquivo `config.json`.
  - Inicializa as classes de processamento de vídeo (`VideoProcessor`) e MQTT (`MqttHandler`).
  - Utiliza o `ThreadPoolExecutor` para executar o processamento de vídeo e o envio de dados MQTT de forma assíncrona.

- **yolo_detector.py**:
  - Usa o OpenCV e o YOLO para detectar passageiros em cada frame do vídeo.
  - Aplica técnicas de **Non-Maximum Suppression** (NMS) para filtrar as detecções.
  - Conta as pessoas que cruzam uma linha imaginária no vídeo, diferenciando passageiros que entram e saem do ônibus.

- **mqtt_handler.py**:
  - Conecta-se ao broker MQTT e publica as contagens de passageiros (entrando e saindo) em tópicos específicos.

# Resultados Esperados

- O script exibirá um vídeo com as pessoas detectadas e contadas, mostrando a quantidade de passageiros entrando e saindo do ônibus.
- As contagens de passageiros serão enviadas para a plataforma IoT via MQTT.


Esse formato inclui todos os detalhes necessários sobre configuração, execução e descrição do código, tornando a documentação clara e fácil de seguir.


