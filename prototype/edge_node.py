from collections import deque
from lru_cache import LRUCache
from lfu_cache import LFUCache
import socket, threading, pickle, keras, json, numpy as np, random, matplotlib.pyplot as plt
from math import e

class SocketThread(threading.Thread):
    def __init__(self, connection, client_info, neighbors_cache:set, count, iteration, buffer_size=1024):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.neighbors_cache = neighbors_cache
        self.count = count
        self.iteration = iteration

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
            if status == 1 and received_data.get("iteration",0) == self.iteration:
                cache = received_data.get("cache",[])
                for data in cache:
                    self.neighbors_cache.add(data)
                msg = pickle.dumps("OK")
                self.connection.sendall(msg)
                self.count()
                break

class EdgeNode:
    def __init__(self, port, neighbors, data_source, strategy = "fql") -> None:
        self.strategy = strategy
        self.init_memory()
        self.frequency = {"total": 0, "hits": 0, "n_hits": 0}
        self.port = port
        self.neighbors = []
        self.active_connections = 0
        self.limit = len(neighbors)
        self.data_source = json.load(open(data_source, 'r'))
        threading.Thread(target=self.init_neighbors).start()
        for n_port in neighbors:
            while True:
                try:
                    n_soc = socket.socket()
                    n_soc.connect(('localhost', n_port))
                    self.neighbors.append(n_soc)
                    break
                except:
                    pass
        while True:
            if len(self.neighbors) >= self.limit and self.active_connections >= self.limit:
                break
        if self.strategy == 'fql':
            self.init_weights()
            self.simulateAndLearn()
        else:
            self.simulation()

    def init_memory(self):
        if self.strategy == 'lru':
            self.memory = LRUCache(2)
        elif self.strategy == 'lfu':
            self.memory = LFUCache(2)
        elif self.strategy == 'fql':
            self.memory = LFUCache(2)
            self.replay_buffer = deque([], maxlen=1000)
            self.init_global()
            self.init_model()
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

    def get_cache(self):
        if self.strategy == "fifo":
            return list(self.memory)
        else :
            return [*self.memory.cache]

    def print_cache(self):
        if self.strategy == "fifo":
            print(self.memory)
        else:
            print(self.memory.cache)

    def receive_data(self, soc):
        received_data = b''
        while str(received_data)[-2] != '.':
            data = soc.recv(1024)
            received_data += data
        received_data = pickle.loads(received_data)
        return received_data
        
    def init_neighbors(self):
        self.soc = socket.socket()
        print("Socket is created.")
        self.soc.bind(("localhost", self.port))
        print("Socket is bound to an address & port number.")
        self.soc.listen(self.limit)
        self.clients = []
        while True:
            try:
                connection, client_info = self.soc.accept()
                print("New Connection from {client_info}.".format(client_info=client_info))
                self.clients.append((connection, client_info))
                self.active_connections+=1
                if self.active_connections >= self.limit:
                    break
            except:
                self.soc.close()
                print("(Timeout) Socket Closed Because no Connections Received.\n")
                break

    def count(self):
        self.received_counter +=1

    def listen_for_neighbors(self, iteration):
        self.neighbors_cache = set()
        self.received_counter = 0
        for connection, client_info in self.clients:
            thread = SocketThread(connection=connection, client_info=client_info, neighbors_cache=self.neighbors_cache, count=self.count, iteration=iteration)
            thread.start()

    def tell_neighbors(self, iteration):
        for soc in self.neighbors:
            while True:
                msg = pickle.dumps({"cache": self.get_cache(), "iteration" : iteration})
                try:
                    soc.sendall(msg)
                    msg = self.receive_data(soc)
                    if msg == "OK": 
                        break
                except socket.error as e:
                    print(e)
        while True:
            if self.received_counter >= self.limit:
                break
 
    def simulation(self):   
        x = []
        y = []
        for i in range(4000):
            self.listen_for_neighbors(i)
            self.tell_neighbors(i)
            reqs = self.data_source[str(i+1)]
            total = len(reqs)
            self.frequency["total"] += total
            data_to_cache = {}
            for req in reqs:
                if self.is_data_cached(req):
                    self.frequency["hits"]+=1
                elif req in self.neighbors_cache:
                    self.frequency["n_hits"]+=1
                else:
                    data_to_cache[req] = data_to_cache.get(req, 0) + 1 

            if len(data_to_cache) > 0:
                data = random.choice(list(data_to_cache.keys()))
                self.cache_data(data)
            
            print("Iteration {} done".format(i+1))
            x.append(i)
            y.append(self.frequency["hits"]/self.frequency["total"])
                           
        print(self.frequency)
        print("Cache hit ratio: {}".format(self.frequency["hits"]/self.frequency["total"]))
        print("Cache hit + n_hit ratio: {}".format((self.frequency["hits"]+self.frequency["n_hits"])/self.frequency["total"]))
        self.print_cache()
        x = np.array(x)
        y = np.array(y)
        plt.plot(x, y, marker='o')
        plt.show()
        self.soc.close()

    def simulateAndLearn(self):
        x = []
        y = []
        epsilon = 1 
        max_epsilon = 1 
        min_epsilon = 0.01 
        decay = 0.01
        for i in range(4000):
            self.listen_for_neighbors(i)
            self.tell_neighbors(i)
            current_state = self.memory.get_state()
            reqs = self.data_source[str(i+1)]
            total = len(reqs)
            self.frequency["total"] += total
            data_to_cache = {}
            for data in reqs:
                self.frequency[data] = self.frequency.get(data, 0) + 1
                if self.is_data_cached(data):
                    self.frequency["hits"]+=1
                    self.remember(current_state, 0, self.get_reward(data, 1), current_state)
                elif data in self.neighbors_cache:
                    self.frequency["n_hits"]+=1
                    self.remember(current_state, 0, self.get_reward(data, 2), current_state)
                else:
                    data_to_cache[data] = data_to_cache.get(data, 0) + 1
            if len(data_to_cache) > 0:
                data = random.choice(list(data_to_cache.keys()))
                if len(self.memory.get_state()) <= self.memory.capacity:
                    self.cache_data(data) 
                    for _ in range(self.frequency[data]):
                        self.remember(current_state, 1, self.get_reward(data, 3), self.memory.get_state())
                else:
                    action = self.chooseAction(current_state, epsilon)
                    if action == 1:
                        self.cache_data(data)
                    for _ in range(self.frequency[data]):
                        self.remember(current_state, action, self.get_reward(data, 3), self.memory.get_state())

            print("Iteration {} done".format(i+1))
            x.append(i)
            y.append(self.frequency["hits"]/self.frequency["total"])
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay * i)
            self.experienceReplay()
            if (i+1) % 20 == 0:
                self.send_weights()
        
        print(self.frequency)
        print("Cache hit ratio: {}".format(self.frequency["hits"]/self.frequency["total"]))
        print("Cache hit + n_hit ratio: {}".format((self.frequency["hits"]+self.frequency["n_hits"])/self.frequency["total"]))
        self.print_cache()
        x = np.array(x)
        y = np.array(y)
        plt.plot(x, y, marker='o')
        plt.show()
        self.soc.close()
    
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
        msg = pickle.dumps(self.model.get_weights())
        self.globalSoc.sendall(msg)
        print('Client send weights to global model')
        received_data = self.receive_data(self.globalSoc)
        weights = received_data.get('weights')
        print('Received weights from the server')
        self.model.set_weights(weights)
        self.target_model.set_weights(weights)
    
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

    def experienceReplay(self, alpha=0.001, gamma=0.9, batch_size=200):
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
