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

    cyclic_barrier = CyclicBarrier(2)

    gradients = []

    def put(self):
        gradient = self.get_argument("gradient")

        self.gradients.append(gradient)
        self.cyclic_barrier.wait()

        #梯度聚合聚合
        print("开始梯度聚合")

