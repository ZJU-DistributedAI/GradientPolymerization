import threading

from server.handler.base import BaseHandler


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

    def put(self):
        gradient = self.get_argument("gradient")

        self.gradients.append(gradient)
        self.cyclic_barrier.wait()

        #梯度聚合聚合
        print("开始梯度聚合")

        data = {}
        self.response(data=data, status_code=200, msg="聚合成功")


    def post(self):
        container_number = int(self.get_argument("containernumber"))
        self.cyclic_barrier = self.CyclicBarrier(10)
        self.response(data={}, status_code=200, msg="设置容器个数成功")