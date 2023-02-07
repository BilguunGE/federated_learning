import random
import json
from numpy import mean
import matplotlib.pyplot as plt
import numpy as np
FILE_A = 0.4
FILE_B = 0.2
FILE_C = 0.1
FILE_D = 0.25
FILE_E = 0.05

def simulateRequests():
    reqList = random.choices(["A","B","C","D","E"], weights=[FILE_A, FILE_B, FILE_C, FILE_D, FILE_E], k=50)
    return reqList

def showResults():
    file = open('results.json', 'r')
    data = json.load(file)

    fifo = data["fifo"] 
    rfu = data["rfu"] 
    lfu = data["lfu"]
    fql = data["fql"]
    #print("fifo: ", mean(fifo))
    #print("rfu: ", mean(rfu))
    #print("lfu: ", mean(lfu))
    #print("fql: ", mean(fql))
    x = np.array(["fifo", "rfu", "lfu", "fql"])
    y = np.array([mean(fifo), mean(rfu), mean(lfu), mean(fql)])
    plt.bar(x,y)
    plt.show()

lru = np.array([])
lfu = np.array([])
fql = np.array([])
for i in range(1,4):
    print(i)
    if i == 1:
        lru = open('lru{}.json'.format(i), 'r')
        lru = json.load(lru)
        lru = np.array(lru["data"])
        lfu = open('lfu{}.json'.format(i), 'r')
        lfu = json.load(lfu)
        lfu = np.array(lfu["data"])
        fql = open('fql{}.json'.format(i), 'r')
        fql = json.load(fql)
        fql = np.array(fql["data"])
    else:
        _lru = open('lru{}.json'.format(i), 'r')
        _lru = json.load(_lru)
        lru += np.array(_lru["data"])
        _lfu = open('lfu{}.json'.format(i), 'r')
        _lfu = json.load(_lfu)
        lfu += np.array(_lfu["data"])
        _fql = open('fql{}.json'.format(i), 'r')
        _fql = json.load(_fql)
        fql += np.array(_fql["data"])
lru /= 3
lfu /= 3
fql /= 3

X = np.array(range(4000))

plt.plot(X, fql, color='r', label='fql')
plt.plot(X, lru, color='g', label='lru')
plt.plot(X, lfu, color='b', label='lfu')

  
# Naming the x-axis, y-axis and the whole graph
plt.xlabel("Time step")
plt.ylabel("Cache hit ratio")
plt.title("FQL vs LFU vs LRU")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()