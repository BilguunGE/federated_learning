import random
import json
FILE_A = 0.4
FILE_B = 0.2
FILE_C = 0.1
FILE_D = 0.25
FILE_E = 0.05

def simulateRequests():
    reqList = random.choices(["A","B","C","D","E"], weights=[FILE_A, FILE_B, FILE_C, FILE_D, FILE_E], k=50)
    return reqList


file = open('node1.json', 'r')
data = json.load(file)
print(data["1"])