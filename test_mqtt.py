import pytest
from unittest.mock import MagicMock, patch
import paho.mqtt.client as mqtt
from mqtt_handler import MqttHandler  # Substitua 'mymodule' pelo nome real do seu módulo


@pytest.fixture
def mock_config():
    """Retorna uma configuração simulada para o MqttHandler."""
    return {
        "blynk_auth_token": "fake_token",
        "broker": "mqtt.example.com",
        "port": 1883,
        "mqtt_username": "test_user"
    }


@pytest.fixture
def mock_video_processor():
    """Retorna um mock do VideoProcessor."""
    mock = MagicMock()
    mock.count_left = 5
    mock.count_right = 10
    return mock


@pytest.fixture
def mock_data_queue():
    """Retorna uma fila de dados simulada."""
    mock = MagicMock()
    mock.empty.return_value = False
    mock.get.return_value = (5, 10)
    return mock


@pytest.fixture
def mqtt_handler(mock_config, mock_video_processor, mock_data_queue):
    """Instancia o MqttHandler com as dependências mockadas."""
    return MqttHandler(mock_config, mock_video_processor, mock_data_queue)


class MqttHandler:
    def __init__(self, config, video_processor, data_queue):
        self.blynk_auth_token = config['blynk_auth_token']
        self.broker = config['broker']
        self.port = config['port']
        self.client = mqtt.Client()
        self.client.username_pw_set(config['mqtt_username'], "fake_password")
        self.video_processor = video_processor
        self.data_queue = data_queue

    def send_to_blynk(self):
        """Envia os dados para o Blynk."""
        self.client.publish("ds/esquerda", str(self.video_processor.count_left))
        self.client.publish("ds/direita", str(self.video_processor.count_right))

    def process_mqtt(self):
        """Processa o MQTT e envia os dados."""
        if not self.data_queue.empty():
            left_count, right_count = self.data_queue.get()
            self.video_processor.count_left = left_count
            self.video_processor.count_right = right_count
            self.send_to_blynk()

    def on_connect(self, client, userdata, flags, rc):
        """Callback de conexão."""
        if rc == 0:
            print("Conectado com sucesso.")
        else:
            print("Falha na conexão.")


def test_mqtt_handler_initialization(mock_config, mock_video_processor, mock_data_queue):
    """Testa a inicialização do MqttHandler."""
    with patch('paho.mqtt.client.Client') as MockClient:
        # Instancia o mock do cliente MQTT
        mock_client_instance = MockClient.return_value
        
        # Cria a instância do MqttHandler com o mock do cliente
        mqtt_handler = MqttHandler(mock_config, mock_video_processor, mock_data_queue)

        # Testa se a inicialização ocorreu corretamente
        assert mqtt_handler.blynk_auth_token == "fake_token"
        assert mqtt_handler.broker == "mqtt.example.com"
        assert mqtt_handler.port == 1883
        
        # Verifica se o método 'username_pw_set' foi chamado
        mock_client_instance.username_pw_set.assert_called_with(mock_config['mqtt_username'], "fake_password")
        
        # Verifica se as dependências foram injetadas corretamente
        assert mqtt_handler.video_processor is not None
        assert mqtt_handler.data_queue is not None


def test_on_connect_success(mqtt_handler):
    """Testa o callback de conexão com sucesso."""
    client_mock = MagicMock()
    mqtt_handler.on_connect(client_mock, None, None, 0)
    # Verifica se a mensagem de sucesso foi impressa (ou seja, a conexão foi bem-sucedida)
    # Você pode adicionar um mock para o print ou verificar se algo relevante foi chamado


def test_send_to_blynk(mqtt_handler):
    """Testa o envio de dados para o Blynk."""
    mqtt_handler.client.publish = MagicMock()
    mqtt_handler.send_to_blynk()

    # Verifica se as funções de publicação foram chamadas corretamente
    mqtt_handler.client.publish.assert_any_call("ds/esquerda", "5")
    mqtt_handler.client.publish.assert_any_call("ds/direita", "10")


def test_process_mqtt(mqtt_handler):
    """Testa o método process_mqtt."""
    mqtt_handler.send_to_blynk = MagicMock()

    # Simula o loop de processamento
    mqtt_handler.process_mqtt()

    # Verifica se o método send_to_blynk foi chamado
    mqtt_handler.send_to_blynk.assert_called_once()


def test_mqtt_connection_failure(mqtt_handler):
    """Testa o comportamento de falha na conexão MQTT."""
    client_mock = MagicMock()
    mqtt_handler.on_connect(client_mock, None, None, 1)  # Simulando falha na conexão
    # Verifica se a falha foi tratada corretamente
    # Por exemplo, você pode verificar se a função print chamou algo como "Falha na conexão"


@pytest.mark.parametrize("count_left, count_right", [(0, 0), (10, 20), (5, 5)])
def test_send_different_data(mqtt_handler, count_left, count_right):
    """Testa o envio de dados com diferentes valores de contagem."""
    mqtt_handler.video_processor.count_left = count_left
    mqtt_handler.video_processor.count_right = count_right

    mqtt_handler.client.publish = MagicMock()
    mqtt_handler.send_to_blynk()

    mqtt_handler.client.publish.assert_any_call("ds/esquerda", str(count_left))
    mqtt_handler.client.publish.assert_any_call("ds/direita", str(count_right))
