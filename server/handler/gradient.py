import threading

from .base import BaseHandler
import json
import numpy as np


class GradientHandler(BaseHandler):
    class CyclicBarrier():
        def __init__(self,count):
            self.condition = threading.Condition()
            self.count = count


        def wait(self):
            try:
                self.condition.acquire()
                self.count -= 1
                if self.count != 0:
                    self.condition.wait()
                else:
                    self.condition.notify_all()
            finally:
                self.condition.release()

    cyclic_barrier = CyclicBarrier(0)

    gradients = []
    container_number = 0

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

    def save_model(name, parameter):
        with open(name, 'w') as json_file:
            json.dump(parameter, json_file, ensure_ascii=False)

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

    def load_json(name):
        with open(name, "r") as load_json:
            return json.load(load_json)

    def put(self):
        gradient1 = self.get_argument("gradient")
        res_sym = self.get_argument("iteration")

        self.gradients.append(gradient1)
        self.cyclic_barrier.wait()


        primModel = self.listToNumpy(self.load_json("./polymerizeModel/resModel{}.json".format(res_sym)))

        aveGrad = self.getAveGrad(self.gradients)
        resModel = self.getResModel(primModel, aveGrad)

        resPath = "./polymerizeModel/resModel" + str(res_sym + 1) + ".json"
        self.save_model(resPath, resModel)

        data = {}
        self.response(data=data, status_code=200, msg="polymerization success")


    def post(self):
        container_number = int(self.get_argument("containernumber"))
        self.cyclic_barrier = self.CyclicBarrier(container_number)
        self.response(data={}, status_code=200, msg="set container number success")

