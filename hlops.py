from tensor import Function

class Dot(Function):
    def forward(self, rhs):
        return self.data @ rhs.data

    def backward(self):
        return self.parents[1].data.T

class Relu(Function):
    def forward(self, x):
        pass

    def backward(self):
        pass

class Softmax(Function):
    def forward(self, x):
        pass

    def backward(self):
        pass

class Softmax_loss(Function):
    def forward(self, x):
        pass

    def backward(self):
        pass
