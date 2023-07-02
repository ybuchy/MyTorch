from tensor import Tensor
import numpy as np

d = Tensor([[1, 2], [-2, 4]], requires_grad=True)

test3 = d.sum()

test3.backward()

print(d.grad)

a = Tensor([[2, 3, 4], [4, 5, 6]], requires_grad=True)
b = Tensor([[3, 4], [5, 6], [7, 8]], requires_grad=True)

test = a.dot(b).sum()

test.backward()

assert a.grad.shape == a.data.shape
assert b.grad.shape == b.data.shape

print(a.grad)
print(b.grad)

exit()

c = Tensor([1, -1], requires_grad=True)

test2 = c.relu().sum()

test2.backward()

assert c.grad.shape == c.data.shape

print(c.grad)

input_layer_weights = Tensor(np.random.rand(3, 4) - .5)
hidden_layer_weights = Tensor(np.random.rand(4, 2) - .5)

input_layer = Tensor(np.random.rand(2, 3) - .5, requires_grad=True)
hidden_layer = input_layer.dot(input_layer_weights)
hidden_layer_new = hidden_layer.relu()
output_layer = hidden_layer_new.dot(hidden_layer_weights)
output_layer_new = output_layer.sum()

hidden_layer_new.backward()

print(input_layer.grad)
