import socket
import pickle
import keras

class EdgeNode:
    def __init__(self, training_data, target_data) -> None:
        self.training_data = training_data
        self.target_data = target_data
        self.soc = socket.socket()
        print('Socket is created.')
        self.soc.connect(("localhost", 10000))
        print('Connected to the server.')
        self.init_model()

    def receive_data(self):
        received_data = b''
        while str(received_data)[-2] != '.':
            data = self.soc.recv(1024)
            received_data += data
        received_data = pickle.loads(received_data)
        return received_data

    def init_model(self):
        msg = 'init_model'
        msg = pickle.dumps(msg)
        self.soc.sendall(msg)
        print('Client requested global model')
        received_data = self.receive_data()
        model = received_data.get('model')
        print('Received model from the server')
        self.model = keras.models.clone_model(model)

    def init_weights(self):
        msg = 'init_weights'
        msg = pickle.dumps(msg)
        self.soc.sendall(msg)
        print('Client requested global model weights')
        received_data = self.receive_data()
        weights = received_data.get('weights')
        print('Received weights from the server')
        self.model.set_weights(weights)

    def train_model(self):
        self.init_weights()
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        self.model.fit(self.training_data, self.target_data, epochs=500, verbose=0)
        msg = pickle.dumps(self.model.get_weights())
        self.soc.sendall(msg)
        print('Client send updated weights to the server')




