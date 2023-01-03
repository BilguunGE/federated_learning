import socket
import pickle
import time
import threading
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

class SocketThread(threading.Thread):
    
    def __init__(self, connection, client_info, buffer_size=1024, recv_timeout=5):
        threading.Thread.__init__(self)
        self.connection = connection
        self.client_info = client_info
        self.buffer_size = buffer_size
        self.recv_timeout = recv_timeout

    def recv(self):
        received_data = b""
        while True:
            try:
                data = self.connection.recv(self.buffer_size)
                received_data += data

                if data == b'': # Nothing received from the client.
                    received_data = b""
                    # If still nothing received for a number of seconds specified by the recv_timeout attribute, return with status 0 to close the connection.
                    if (time.time() - self.recv_start_time) > self.recv_timeout:
                        return None, 0 # 0 means the connection is no longer active and it should be closed.
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
                else:
                    # In case data are received from the client, update the recv_start_time to the current time to reset the timeout counter.
                    self.recv_start_time = time.time()
    
            except BaseException as e:
                print("Error Receiving Data from the Client: {msg}.\n".format(msg=e))
                return None, 0

    def run(self):
        while True:
            self.recv_start_time = time.time()
            time_struct = time.gmtime()
            date_time = "Waiting to Receive Data Starting from {day}/{month}/{year} {hour}:{minute}:{second} GMT".format(year=time_struct.tm_year, month=time_struct.tm_mon, day=time_struct.tm_mday, hour=time_struct.tm_hour, minute=time_struct.tm_min, second=time_struct.tm_sec)
            print(date_time)
            received_data, status = self.recv()
            if status == 0:
                self.connection.close()
                print("Connection Closed with {client_info} either due to inactivity for {recv_timeout} seconds or due to an error.".format(client_info=self.client_info, recv_timeout=self.recv_timeout), end="\n\n")
                break
            if status == 1 and received_data == 'init_model':
                msg = {'model':model}
                msg = pickle.dumps(msg)
                self.connection.sendall(msg)
                print("Server sent a model to the client.")
            elif status == 1 and received_data == 'init_weights':
                msg = {'weights':model.get_weights()}
                msg = pickle.dumps(msg)
                self.connection.sendall(msg)
                print("Server sent weights to the client.", self.client_info)
            elif status == 1:
                new_weights = np.array(received_data, dtype=object)
                weights.append(new_weights)
                msg = {'weights':model.get_weights()}
                msg = pickle.dumps(msg)
                self.connection.sendall(msg)
                break

training_data = np.array([[0,0], [0,1], [1,0], [1,1]])
target_data = np.array([[0],[1],[1],[0]])
model = Sequential()
model.add(Dense(3, input_dim=2, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam')

soc = socket.socket()
print("Socket is created.")

soc.bind(("localhost", 10000))
print("Socket is bound to an address & port number.")

soc.listen(1)
print("Listening for incoming connection ...")

num_of_clients = 2
clients = []
while True:
    try:
        connection, client_info = soc.accept()
        print("New Connection from {client_info}.".format(client_info=client_info))
        clients.append([connection, client_info])
    except:
        soc.close()
        print("(Timeout) Socket Closed Because no Connections Received.\n")
        break
    if len(clients) == num_of_clients:
        break

num_of_iterations = 0
while True:
    num_of_iterations+=1
    threads = []
    weights = []
    prediction = model.predict(training_data, verbose=0)
    error = np.sum(np.abs(prediction-target_data))
    if num_of_iterations % 1 == 0:
        print('---------------------------------')
        print('error: ', error)
        print('---------------------------------')
    if error < 0.01:
        print("Model successfully trained with {i} iterations".format(i=num_of_iterations))
        break
    for client in clients:
        [connection, client_info] = client
        socket_thread = SocketThread(connection=connection,
                                    client_info=client_info, 
                                    buffer_size=1024,
                                    recv_timeout=10)
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

soc.close()
