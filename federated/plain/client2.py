from edge_node import EdgeNode
import numpy as np

batch_data = np.array([[1,0], [1,1]])
target_data = np.array([[1], [0]])
client2 = EdgeNode(batch_data, target_data)

while True:
    client2.train_model()
