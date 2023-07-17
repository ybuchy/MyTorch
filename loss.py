from tensor import Tensor
import numpy as np

class Softmax_loss():
    def __call__(self, tensor, labels):
        self.tensor = tensor
        self.labels = labels
        # numerical stabilization
        m = tensor - tensor.amax(axis=0)
        e_m = m.exp()
        self.activated = e_m / e_m.sum(axis=0)
        # TODO: (testing)
        idk = -(labels * self.activated.log()).sum(axis=1)
        return sum(idk) / idk.size

    def backward(self):
        grad = self.activated - self.labels
        self.tensor.backward(grad)
