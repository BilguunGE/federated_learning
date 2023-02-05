import random
import json
from numpy import mean
FILE_A = 0.4
FILE_B = 0.2
FILE_C = 0.1
FILE_D = 0.25
FILE_E = 0.05

def simulateRequests():
    reqList = random.choices(["A","B","C","D","E"], weights=[FILE_A, FILE_B, FILE_C, FILE_D, FILE_E], k=50)
    return reqList


file = open('results.json', 'r')
data = json.load(file)

fifo = data["fifo"] 
rfu = data["rfu"] 
lfu = data["lfu"]

print("fifo: ", mean(fifo))
print("rfu: ", mean(rfu))
print("lfu: ", mean(lfu))