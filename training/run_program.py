import os
import threading
import json
import time
from .simple_layer import *


# creat thread
class myThread(threading.Thread):
    def __init__(self, threadID, iteration):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.iteration = iteration

    def run(self):
        runProgram(self.threadID, self.iteration)

# runs in thread
def runProgram(threadID, iteration):
    os.system("python GradientPolymerization.py " + str(threadID) + " " + str(iteration))

def load_json(name):
    with open(name, "r") as load_json:
        return json.load(load_json)


def listToNumpy(parameters):
    W1 = np.array(parameters["W1"])
    b1 = np.array(parameters["b1"])
    W2 = np.array(parameters["W2"])
    b2 = np.array(parameters["b2"])
    W3 = np.array(parameters["W3"])
    b3 = np.array(parameters["b3"])
    parameters = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2,
        'W3': W3,
        'b3': b3
    }
    return parameters


def getResModel(primModel, grad):
    dW1 = grad["W1"]
    db1 = grad["b1"]
    dW2 = grad["W2"]
    db2 = grad["b2"]
    dW3 = grad["W3"]
    db3 = grad["b3"]
    W1 = primModel["W1"]
    b1 = primModel["b1"]
    W2 = primModel["W2"]
    b2 = primModel["b2"]
    W3 = primModel["W3"]
    b3 = primModel["b3"]
    parameters = {
        'W1': (W1 - dW1).tolist(),
        'b1': (b1 - db1).tolist(),
        'W2': (W2 - dW2).tolist(),
        'b2': (b2 - db2).tolist(),
        'W3': (W3 - dW3).tolist(),
        'b3': (b3 - db3).tolist()
    }

    return parameters


def getAveGrad(grads):
    # get zero matrix
    resW1 = grads[0]['W1'] - grads[0]['W1']
    resb1 = grads[0]['b1'] - grads[0]['b1']
    resW2 = grads[0]['W2'] - grads[0]['W2']
    resb2 = grads[0]['b2'] - grads[0]['b2']
    resW3 = grads[0]['W3'] - grads[0]['W3']
    resb3 = grads[0]['b3'] - grads[0]['b3']
    lenOfGrads = len(grads)
    for grad in grads:
        resW1 = resW1 + grad['W1'] / lenOfGrads
        resb1 = resb1 + grad['b1'] / lenOfGrads
        resW2 = resW2 + grad['W2'] / lenOfGrads
        resb2 = resb2 + grad['b2'] / lenOfGrads
        resW3 = resW3 + grad['W3'] / lenOfGrads
        resb3 = resb3 + grad['b3'] / lenOfGrads
    parameters = {
        'W1': resW1,
        'b1': resb1,
        'W2': resW2,
        'b2': resb2,
        'W3': resW3,
        'b3': resb3
    }
    return parameters


def save_model(name, parameter):
    with open(name, 'w') as json_file:
        json.dump(parameter, json_file, ensure_ascii=False)


if __name__ == '__main__':

    # num of batch is 600, need 10 times to finish all img
    mainThreadTime = 0
    startTime = time.time()
    for iteration in range(10):
        threads = []
        # run program by multithreading method
        for ID in range(10):
            thread = myThread(ID, iteration)
            threads.append(thread)

        for thread in threads:
            thread.start()

        # main thread waits until all subThread finished
        for t in threads:
            t.join()

        print("Polymerize parameters")
        primModel = listToNumpy(load_json("./polymerizeModel/resModel{}.json".format(iteration)))

        grads = []
        for i in range(10):
            grad = listToNumpy(load_json("./result/grad" + str(i) + ".json"))
            grads.append(grad)

        aveGrad = getAveGrad(grads)
        resModel = getResModel(primModel, aveGrad)

        resPath = "./polymerizeModel/resModel" + str(iteration + 1) + ".json"
        save_model(resPath, resModel)
    endTime = time.time()
    mainThreadTime = endTime - startTime
    print("main time:{}".format(mainThreadTime))