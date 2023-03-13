import socket, pickle, threading, numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

class SocketThread(threading.Thread):
    def __init__(self, connection, client_info, buffer_size=1024):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size

    def recv(self):
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data

                if data == b'': # Nothing received from the client.
                    received_data = b""
                elif str(data)[-2] == '.':
                    print("All data ({data_len} bytes) Received from {client_info}.".format(client_info=self.client_info, data_len=len(received_data)))
                    if len(received_data) > 0:
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
            if status == 1 and received_data == 'init_model':
                msg = pickle.dumps({'model':model})
                self.connection.sendall(msg)
                print("Server sent a model to the client.")
            elif status == 1 and received_data == 'init_weights':
                msg = pickle.dumps({'weights':model.get_weights()})
                self.connection.sendall(msg)
                print("Server sent weights to the client.", self.client_info)
            elif status == 1:
                new_weights = np.array(received_data, dtype=object)
                weights.append(new_weights)
                #msg = pickle.dumps({'weights':model.get_weights()})
                #self.connection.sendall(msg)
                break

#training_data = np.array([[0,0], [0,1], [1,0], [1,1]])
#target_data = np.array([[0],[1],[1],[0]])
model = Sequential()
model.add(Dense(5, input_dim=2, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

soc = socket.socket()
print("Socket is created.")

soc.bind(("localhost", 10000))
print("Socket is bound to an address & port number.")

soc.listen(3)
print("Listening for incoming connection ...")

num_of_clients = 3
clients = []
while True:
    try:
        connection, client_info = soc.accept()
        print("New Connection from {client_info}.".format(client_info=client_info))
        clients.append([connection, client_info])
    except:
        soc.close()
        print("Socket Closed Because no Connections Received.\n")
        break
    if len(clients) == num_of_clients:
        break

num_of_iterations = 0
while num_of_iterations <= 4000:
    num_of_iterations+=1
    threads = []
    weights = []
    #prediction = model.predict(training_data, verbose=0)
    #error = np.sum(np.abs(prediction-target_data))
    #if num_of_iterations % 5 == 0:
    #    print('---------------------------------')
    #    print('error: ', error)
    #    print('---------------------------------')
    #if error < 0.01:
    #    print("Model successfully trained with {i} iterations".format(i=num_of_iterations))
    #    break
    for client in clients:
        [connection, client_info] = client
        socket_thread = SocketThread(connection=connection,
                                    client_info=client_info, 
                                    buffer_size=1024)
        threads.append(socket_thread)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    updated_weights = None
    for weight in weights:
        if updated_weights is None:
            updated_weights = weight
        else:
            updated_weights = np.add(updated_weights, weight)
    updated_weights /= len(clients)
    model.set_weights(updated_weights)
    for client in clients:
        [connection, client_info] = client
        msg = pickle.dumps({'weights':updated_weights})
        connection.sendall(msg)
        print("Server sent updated weights to the client {}.".format(client_info))

soc.close()
