from edge_node import EdgeNode
import numpy as np

batch_data = np.array([[0,0], [0,1]])
target_data = np.array([[0], [1]])
client1 = EdgeNode(batch_data, target_data)

while True:
    client1.train_model()
