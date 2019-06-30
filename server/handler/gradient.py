import threading

from server.handler.base import BaseHandler
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
        gradient = self.get_argument("gradient")

        self.gradients.append(gradient)
        self.cyclic_barrier.wait()

        #梯度聚合聚合
        print("开始梯度聚合")
        # 获取梯度
        for i in range(10):
            grad = self.listToNumpy(self.load_json("./result/grad" + str(i) + ".json"))
            grads.append(grad)

        # 加载上一轮迭代完的模型
        primModel = self.listToNumpy(self.load_json("./polymerizeModel/resModel{}.json".format(iteration)))  # 路径还有问题

        aveGrad = self.getAveGrad(grads)
        resModel = self.getResModel(primModel, aveGrad)
        # 每次梯度聚合后更新模型并保存
        resPath = "./polymerizeModel/resModel" + str(iteration + 1) + ".json"  # 这里的iteration表示第几次聚合
        self.save_model(resPath, resModel)

        data = {}
        self.response(data=data, status_code=200, msg="聚合成功")


    def post(self):
        container_number = int(self.get_argument("containernumber"))
        self.cyclic_barrier = self.CyclicBarrier(10)
        self.response(data={}, status_code=200, msg="设置容器个数成功")

