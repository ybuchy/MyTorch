from __future__ import annotations
from typing import Union, Optional
import numpy as np


class Tensor:
    # cur IDEA: if you have a tensor and do some calculation, then a new tensor will be created with a backward_function that has the parent tensors and how the gradient of the function is calculated (~> NO in-place operations for Tensors with requires_grad=True)
    def __init__(self, data: Union[int, float, list, np.ndarray], requires_grad: Optional[bool]=None, back_fn: Function=None):
        self.requires_grad = requires_grad

        if isinstance(data, (int, float, np.int64, np.int32, np.float32, np.float64)):
            self.data = np.array([data])

        elif isinstance(data, list):
            self.data = np.array(data)

        else:
            assert type(data) == np.ndarray, "wrong input data type"
            self.data = data

        self.grad = np.zeros_like(self.data)
        self.back_fn = back_fn



    """
    # TODO
    @property
    def grad_tracked(self):
        pass
    """

    # TODO
    def sum(self) -> Tensor:
        pass

    def backward(self, grad: Optional[Tensor]=None) -> None:
        if grad is None:
            # TODO: check for shape of data
            grad = np.array([1])
        if self.back_fn is None: # Tensor is leaf of computational graph -> save grad
            # TODO @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            weight = np.zeros_like(self.grad)
            self.grad += weight * grad
            return
        if not self.back_fn.requires_grad: return

        for parent in self.back_fn.parents:
            if not parent.requires_grad: continue
            # REDO ~ this is wasting much memory - use toposort for that!
            
            # TODO @@@@@@@@@@@@@@@@@@@@@@@22
            parent.grad += np.zeros_like(parent.grad)
            # TODO: problem - long recursion for long calculations!
            parent.backward(self.back_fn.backward(grad))
        
    # hlops
    def dot(self, tensor: Tensor) -> Tensor: return Dot(self, tensor).apply()
    def relu(self) -> Tensor: return Relu(self).apply()
    def __add__(self, rhs: Tensor) -> Tensor: return Add(self, rhs).apply()
    def __matmul__(self, rhs: Tensor) -> Tensor: return self.dot(rhs)

    def __str__(self): return f"Tensor({np.array2string(self.data)})\n"

class Function:
    def __init__(self, tensor: Tensor, *tensors: Sequence[Tensor]):
        self.parents = [tensor, *tensors]
        self.requires_grad = any(parent.requires_grad for parent in self.parents)

    def forward(self):
        raise NotImplementedError(f"forward function of {type(self)} not implemented")

    def backward(self):
        raise NotImplementedError(f"backward function of {type(self)} not implemented")

    def apply(self):
        if not self.requires_grad: return
        return Tensor(self.forward(*self.parents), requires_grad=self.requires_grad, back_fn=self)

class Dot(Function):
    def forward(self, lhs, rhs):
        return lhs.data @ rhs.data

    def backward(self, grad):
        return self.parents[1].data.T * grad

class Relu(Function):
    def forward(self, tensor):
        return np.maximum(0, tensor.data)
    
    def backward(self, grad):
        data = self.parents[0].data
        return np.maximum(np.sign(data), np.zeros_like(data))
