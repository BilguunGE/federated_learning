from collections import deque
from lru_cache import LRUCache
from lfu_cache import LFUCache
import socket, threading, pickle, keras, json, numpy as np, random
from math import e

class SocketThread(threading.Thread):
    def __init__(self, connection, client_info, is_data_cached, buffer_size=1024):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.is_data_cached = is_data_cached

    def recv(self):
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data

                if data == b'': # Nothing received from the client.
                    received_data = b""
                elif str(data)[-2] == '.':
                    if len(received_data) > 0:
                        #print("All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data)))
                        try:
                            # Decoding the data (bytes).
                            received_data = pickle.loads(received_data)
                            # Returning the decoded data.
                            return received_data, 1
    
                        except BaseException as e:
                            print("Error Decoding the Client's Data: {msg}.\n".format(msg=e))
                            return None, 0
    
            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def run(self):
        while True:
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print("Connection Closed with {client_info} due to an error.".format(client_info=self.client_info), end="\n\n")
                break
            if status == 1:
                data_to_look = received_data
                if self.is_data_cached(data_to_look):
                    msg = pickle.dumps(data_to_look)
                else:
                    msg = pickle.dumps('0')
                self.connection.sendall(msg)


class EdgeNode:
    def __init__(self, port, neighbors, data_source, strategy = "fifo") -> None:
        self.strategy = strategy
        self.init_memory()
        self.frequency = {"total": 0, "hits":0, "n_hits":0}
        self.port = port
        self.neighbors = []
        self.active_connections = 0
        self.limit = len(neighbors)
        file = open(data_source, 'r')
        self.data_source = json.load(file)
        threading.Thread(target=self.listen_for_neighbors).start()
        for n_port in neighbors:
            while True:
                try:
                    n_soc = socket.socket()
                    n_soc.connect(('localhost', n_port))
                    self.neighbors.append(n_soc)
                    break
                except:
                    pass
        if self.strategy == 'fql':
            self.init_weights()
            self.simulateAndLearn()
        else:
            self.simulation()

    def init_memory(self):
        if self.strategy == "lru":
            self.memory = LRUCache(2)
        elif self.strategy == 'lfu':
            self.memory = LFUCache(2)
        elif self.strategy == 'fql':
            self.memory = LFUCache(2)
            self.replay_buffer = deque([], maxlen=1000)
            self.init_global()
            self.init_model()
            self.replay_rounds = 0
        else:
            self.memory = deque([], 2)

    def cache_data(self, data):
        if self.strategy == "fifo":
            self.memory.appendleft(data)
        else:
            self.memory.set(data, 1)

    def is_data_cached(self, data):
        if self.strategy == "fifo":
            return data in self.memory  
        else:
            return self.memory.get(data) != -1

    def print_cache(self):
        if self.strategy == "fifo":
            print(self.memory)
        else:
            print(self.memory.cache)

    def init_global(self):
        self.globalSoc = socket.socket()
        print('Socket is created.')
        self.globalSoc.connect(("localhost", 10000))
        print('Connected to the global server.')

    def init_model(self):
        msg = 'init_model'
        msg = pickle.dumps(msg)
        self.globalSoc.sendall(msg)
        print('Client requested global model')
        received_data = self.receive_data(self.globalSoc)
        model = received_data.get('model')
        print('Received model from the server')
        self.model = keras.models.clone_model(model)
        self.target_model = keras.models.clone_model(model)
        self.model.compile(loss='mse', optimizer='adam')



    def init_weights(self):
        msg = 'init_weights'
        msg = pickle.dumps(msg)
        self.globalSoc.sendall(msg)
        print('Client requested global model weights')
        received_data = self.receive_data(self.globalSoc)
        weights = received_data.get('weights')
        print('Received weights from the server')
        self.model.set_weights(weights)
        self.target_model.set_weights(weights)
    
    def send_weights(self):
        msg = pickle.dumps(self.target_model.get_weights())
        self.globalSoc.sendall(msg)
        print('Client send weights to global model')
        received_data = self.receive_data(self.globalSoc)
        weights = received_data.get('weights')
        print('Received weights from the server')
        self.model.set_weights(weights)
        self.target_model.set_weights(weights)
        self.simulateAndLearn()
    
    def chooseAction(self, state, epsilon = 0.9):
        if random.random() < epsilon:
            return random.choice([0, 1])
        else:
            return self.chooseMax(state)
    
    def chooseMax(self, state):
        prediction = self.target_model.predict(np.array([state]), verbose=0)
        return np.argmax(prediction[0])

    def get_reward(self, data, method):
        popularity = (self.frequency[data]/self.frequency["total"])
        if method == 1:
            return popularity * e**(-0.2*10)
        elif method ==2:
            return popularity * e**((-0.2*10)+(-0.3*20))
        else: 
            return popularity * e**((-0.2*10)+(-0.5*200))

    def remember(self, current_state, action, reward, next_state):
        current_state = np.array(current_state)
        next_state = np.array(next_state)
        self.replay_buffer.appendleft((current_state, action, reward, next_state))

    def experienceReplay(self, alpha=0.1, gamma=0.8, batch_size=200):
        if len(self.replay_buffer) < 600:
            return
        mini_batch = random.sample(self.replay_buffer, batch_size)
        current_states = np.array([states[0] for states in mini_batch])
        current_qs = self.model.predict(current_states, verbose=0)
        next_states = np.array([states[3] for states in mini_batch])
        future_qs = self.target_model.predict(next_states, verbose=0)

        X = []
        Y = []

        for index, (state, action, reward, _) in enumerate(mini_batch):
            max_future_q = reward + gamma * np.max(future_qs[index])
            current_q = current_qs[index]
            current_q[action] = (1-alpha) * current_q[action] + alpha * max_future_q
            X.append(state)
            Y.append(current_q)
        self.model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)
        self.replay_rounds += 1
        if self.replay_rounds >= 20:
            self.target_model.set_weights(self.model.get_weights())
            self.replay_rounds = 0

    def listen_for_neighbors(self):
        soc = socket.socket()
        print("Socket is created.")
        soc.bind(("localhost", self.port))
        print("Socket is bound to an address & port number.")
        soc.listen(self.limit)
        while True:
            try:
                connection, client_info = soc.accept()
                print("New Connection from {client_info}.".format(client_info=client_info))
                client = SocketThread(connection=connection, client_info=client_info, is_data_cached=self.is_data_cached)
                client.start()
                self.active_connections+=1
            except:
                soc.close()
                print("(Timeout) Socket Closed Because no Connections Received.\n")
                break

    def simulation(self):
        while True:
            if self.active_connections >= self.limit:
                break
        for i in range(4000):
            reqs = self.data_source[str(i+1)]
            total = len(reqs)
            self.frequency["total"] += total
            data_to_cache = set()
            data_freq = {}
            for req in reqs:
                data_freq[req] = data_freq.get(req, 0) + 1
                if self.is_data_cached(req):
                    self.frequency["hits"]+=1
                else:
                    data_to_cache.add(req)
            for data in data_to_cache:
                if self.ask_neighbors(data):
                    self.frequency["n_hits"]+=data_freq[data]
                else:
                    self.cache_data(data) 
            print("Iteration {} done".format(i+1))
                           
        print(self.frequency)
        print("Cache hit ratio: {}".format(self.frequency["hits"]/self.frequency["total"]))
        print("Cache hit + n_hit ratio: {}".format((self.frequency["hits"]+self.frequency["n_hits"])/self.frequency["total"]))
        self.print_cache()

    def simulateAndLearn(self):
        while True:
            if self.active_connections >= self.limit:
                break
        epsilon = 1 
        max_epsilon = 1 
        min_epsilon = 0.01 
        decay = 0.01
        for i in range(4000):
            current_state = self.memory.get_state()
            reqs = self.data_source[str(i+1)]
            total = len(reqs)
            self.frequency["total"] += total
            data_to_cache = set()
            data_req_freq = {}
            for data in reqs:
                data_req_freq[data] = data_req_freq.get(data, 0) + 1
                self.frequency[data] = self.frequency.get(data, 0) + 1
                if self.is_data_cached(data):
                    self.frequency["hits"]+=1
                    #self.remember(current_state, 0, self.get_reward(data, 1), current_state)
                else:
                    data_to_cache.add(data)
            for data in data_to_cache:
                if self.ask_neighbors(data):
                    self.frequency["n_hits"]+=data_req_freq[data]
                    #for _ in range(data_req_freq[data]):
                        #self.remember(current_state, 0, self.get_reward(data, 2), current_state)
                else:
                    if len(self.memory.get_state()) < self.memory.capacity:
                        self.cache_data(data) 
                        #for _ in range(self.frequency[data]):
                        #    self.remember(current_state, 1, self.get_reward(data, 3), self.memory.get_state())

                    else:
                        action = self.chooseAction(current_state, epsilon)
                        if action == 1:
                            self.cache_data(data)
                        for _ in range(self.frequency[data]):
                            self.remember(current_state, action, self.get_reward(data, 3), self.memory.get_state())

            print("Iteration {} done".format(i+1))
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * i)
            self.experienceReplay()            
        print(self.frequency)
        print("Cache hit ratio: {}".format(self.frequency["hits"]/self.frequency["total"]))
        print("Cache hit + n_hit ratio: {}".format((self.frequency["hits"]+self.frequency["n_hits"])/self.frequency["total"]))
        self.print_cache()
        self.send_weights()

    def ask_neighbors(self, data_request):
        for soc in self.neighbors:
            msg = pickle.dumps(data_request)
            try:
                soc.sendall(msg)
            except socket.error as e:
                print(e)
            received_data = self.receive_data(soc)
            if received_data != '0':
                return True
        return False

    def receive_data(self, soc):
        received_data = b''
        while str(received_data)[-2] != '.':
            data = soc.recv(1024)
            received_data += data
        received_data = pickle.loads(received_data)
        return received_data






