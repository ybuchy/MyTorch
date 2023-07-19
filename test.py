from tensor import Tensor
from loss import Softmax_loss
import torch
from torch import nn
import numpy as np
import unittest

class Test(unittest.TestCase):
    # TODO
    def test_unary_grad(self):
        pass

    # TODO
    def test_binary_grad(self):
        pass

    def test_div_grad(self):
        data = np.random.rand(10, 20)
        const = np.random.rand()
        my_tensor = Tensor(data, requires_grad=True)
        torch_tensor = torch.tensor(data, requires_grad=True)

        (my_tensor / const).sum().backward()
        (torch_tensor / const).sum().backward()

        np.testing.assert_almost_equal(my_tensor.grad, torch_tensor.grad)        

    def test_add_grad(self):
        data1 = np.random.rand(10, 20)
        data2 = np.random.rand(10, 20)
        my_tensor1 = Tensor(data1, requires_grad=True)
        my_tensor2 = Tensor(data2, requires_grad=True)
        torch_tensor1 = torch.tensor(data1, requires_grad=True)
        torch_tensor2 = torch.tensor(data2, requires_grad=True)

        (my_tensor1 + my_tensor2).sum().backward()
        (torch_tensor1 + torch_tensor2).sum().backward()

        np.testing.assert_almost_equal(my_tensor1.grad, torch_tensor1.grad)
        np.testing.assert_almost_equal(my_tensor2.grad, torch_tensor2.grad)

    def test_sub_grad(self):
        data1 = np.random.rand(10, 20)
        data2 = np.random.rand(10, 20)
        my_tensor1 = Tensor(data1, requires_grad=True)
        my_tensor2 = Tensor(data2, requires_grad=True)
        torch_tensor1 = torch.tensor(data1, requires_grad=True)
        torch_tensor2 = torch.tensor(data2, requires_grad=True)

        (my_tensor1 - my_tensor2).sum().backward()
        (torch_tensor1 - torch_tensor2).sum().backward()

        np.testing.assert_almost_equal(my_tensor1.grad, torch_tensor1.grad)
        np.testing.assert_almost_equal(my_tensor2.grad, torch_tensor2.grad)

    def test_sum_grad(self):
        data = np.random.rand(10, 20)
        data2 = np.random.rand(10, 20, 30, 40, 50)
        my_tensor = Tensor(data, requires_grad=True)
        torch_tensor = torch.tensor(data, requires_grad=True)
        my_tensor2 = Tensor(data2, requires_grad=True)
        torch_tensor2 = torch.tensor(data2, requires_grad=True)

        my_tensor.sum().backward()
        torch_tensor.sum().backward()
        my_tensor2.sum(axis=3).sum().backward()
        torch_tensor2.sum(axis=3).sum().backward()

        np.testing.assert_almost_equal(my_tensor.grad, torch_tensor.grad)
        np.testing.assert_almost_equal(my_tensor2.grad, torch_tensor2.grad)

    def test_amax_grad(self):
        data = np.random.rand(20, 30, 40, 50)
        my_tensor = Tensor(data, requires_grad=True)
        torch_tensor = torch.tensor(data, requires_grad=True)
        my_tensor2 = Tensor(data, requires_grad=True)
        torch_tensor2 = torch.tensor(data, requires_grad=True)

        torch_tensor.amax().sum().backward()
        my_tensor.amax().sum().backward()
        torch_tensor2.amax(axis=2).sum().backward()
        my_tensor2.amax(axis=2).sum().backward()

        np.testing.assert_almost_equal(my_tensor.grad, torch_tensor.grad)
        np.testing.assert_almost_equal(my_tensor2.grad, torch_tensor2.grad)

    def test_exp_grad(self):
        data = np.random.rand(10, 20)
        my_tensor = Tensor(data, requires_grad=True)
        torch_tensor = torch.tensor(data, requires_grad=True)

        my_tensor.exp().sum().backward()
        torch_tensor.exp().sum().backward()
        
        np.testing.assert_almost_equal(my_tensor.grad, torch_tensor.grad)

    def test_log_grad(self):
        data = np.random.rand(10, 20)
        my_tensor = Tensor(data, requires_grad=True)
        torch_tensor = torch.tensor(data, requires_grad=True)

        my_tensor.log().sum().backward()
        torch_tensor.log().sum().backward()
        
        np.testing.assert_almost_equal(my_tensor.grad, torch_tensor.grad)

    def test_dot_grad(self):
        data_a = np.random.rand(10, 20)
        data_b = np.random.rand(20, 30)
        tensor_a = Tensor(data_a, requires_grad=True)
        torch_a = torch.tensor(data_a, requires_grad=True)
        tensor_b = Tensor(data_b, requires_grad=True)
        torch_b = torch.tensor(data_b, requires_grad=True)

        tensor_a.dot(tensor_b).sum().backward()
        (torch_a @ torch_b).sum().backward()

        np.testing.assert_almost_equal(tensor_a.grad, torch_a.grad)
        np.testing.assert_almost_equal(tensor_b.grad, torch_b.grad)

    def test_relu_grad(self):
        data = np.array([[1, -1, 0], [0, 1, -1]], float)
        my_tensor = Tensor(data, requires_grad=True)
        torch_tensor = torch.tensor(data, requires_grad=True)

        my_tensor.relu().sum().backward()
        torch_tensor.relu().sum().backward()        

        np.testing.assert_almost_equal(my_tensor.grad, torch_tensor.grad)

    def test_nn(self):
        data_ilw = np.random.rand(3, 4) - .5
        data_olw = np.random.rand(4, 2) - .5
        data_il = np.random.rand(2, 3) - .5
        my_input_layer_weights = Tensor(data_ilw, requires_grad=True)
        torch_input_layer_weights = torch.tensor(data_ilw, requires_grad=True)
        my_hidden_layer_weights = Tensor(data_olw, requires_grad=True)
        torch_hidden_layer_weights = torch.tensor(data_olw, requires_grad=True)

        # redo with nn.linear?
        Tensor(data_il).dot(my_input_layer_weights).relu().dot(my_hidden_layer_weights).sum().backward()
        a = (torch.tensor(data_il) @ torch_input_layer_weights).relu()
        (a @ torch_hidden_layer_weights).sum().backward()

        np.testing.assert_almost_equal(my_input_layer_weights.grad, torch_input_layer_weights.grad)
        np.testing.assert_almost_equal(my_hidden_layer_weights.grad, torch_hidden_layer_weights.grad)

    def test_softmax_loss(self):
        my_loss = Softmax_loss()
        torch_loss = nn.CrossEntropyLoss()
        data = np.random.rand(20, 20)
        data2 = np.random.rand(20, 20)
        my_out = Tensor(data, requires_grad=True)
        my_labels = Tensor(data2)
        torch_out = torch.tensor(data, requires_grad=True)
        torch_labels = torch.tensor(data2)

        loss1 = my_loss(my_out, my_labels)
        loss2 = torch_loss(torch_out, torch_labels)

        np.testing.assert_almost_equal(loss1.data, loss2.numpy())

    def test_softmax_loss_grad(self):
        pass
