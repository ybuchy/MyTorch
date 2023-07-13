from __future__ import annotations
from enum import Enum, auto
from typing import Union, Optional
import numpy as np


class Tensor:
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
    def zeros_like():
        pass

    # REDO ~ this is using much more memory than it needs - use toposort for that!
    def backward(self, grad: Optional[Tensor]=None) -> None:
        if grad is None:
            # TODO: check for shape of data
            grad = np.array([1])
        self.grad += grad
        if self.back_fn is None: # Tensor is leaf of computational graph -> save grad
            # TODO
            return
        if not self.back_fn.requires_grad: return

        if self.back_fn.type in [ftype.unary, ftype.reduce]:
            parent = self.back_fn.parents[0]
            grad = self.back_fn.backward(grad)
            parent.backward(grad)
        elif self.back_fn.type == ftype.binary:
            grad1, grad2 = self.back_fn.backward(grad)
            self.back_fn.parents[0].backward(grad1)
            self.back_fn.parents[1].backward(grad2)

        
    # hlops
    def dot(self, tensor: Tensor) -> Tensor: return Dot(self, tensor).apply()
    def relu(self) -> Tensor: return Relu(self).apply()
    def sum(self) -> Tensor: return Sum(self).apply()
    def exp(self) -> Tensor: return Exp(self).apply()
    def log(self) -> Tensor: return Log(self).apply()
    def amax(self, axis=None) -> Tensor: return Amax(self, axis).apply()
    def softmax_loss(self, labels) -> Tensor: return Softmax_loss(self, labels).apply()
    def __add__(self, rhs: Tensor) -> Tensor: return Add(self, rhs).apply()
    def __neg__(self) -> Tensor: return Neg(self).apply()
    def __sub__(self, rhs: Tensor) -> Tensor: return Add(self, -rhs).apply()
    def __matmul__(self, rhs: Tensor) -> Tensor: return self.dot(rhs)
    def __truediv__(self, rhs: int | float) -> Tensor: return Div(self, rhs).apply()

    def __str__(self): return f"Tensor({np.array2string(self.data)})\n"


class ftype(Enum):
    unary = auto()
    binary = auto()
    reduce = auto()


class Function:
    type: ftype

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


class Add(Function):
    type = ftype.binary

    def forward(self, lhs, rhs):
        return lhs.data + rhs.data

    def backward(self, grad):
        return (grad, grad)


class Neg(Function):
    type = ftype.unary

    def forward(self, tensor):
        return -tensor.data

    def backward(self, grad):
        return -grad


class Div(Function):
    # TODO rly unary? what about tensor / tensor
    type = ftype.unary

    # TODO const / tensor diff
    def forward(self, lhs: Tensor, const: int | float):
        self.const = const
        return lhs.data / const

    def backward(self, grad):
        return (1 / self.const) * grad


class Dot(Function):
    type = ftype.binary

    def forward(self, lhs, rhs):
        return lhs.data @ rhs.data

    def backward(self, grad):
        fst = grad @ self.parents[1].data.T
        scnd = self.parents[0].data.T @ grad
        return (fst, scnd)


class Amax(Function):
    type = ftype.reduce

    def forward(self, tensor, axis):
        return np.amax(tensor.data, axis)

    # TODO
    def backward(self):
        pass


class Exp(Function):
    type = ftype.unary

    def forward(self, tensor):
        self.exp = np.exp(tensor.data)
        return self.exp

    def backward(self, grad):
        return grad * self.exp


class Log(Function):
    type = ftype.unary

    def forward(self, tensor):
        self.tensor = tensor
        return np.log(tensor.data)

    def backward(self, grad):
        return 1 / self.tensor.data * grad


class Relu(Function):
    type = ftype.unary

    def forward(self, tensor):
        return np.maximum(0, tensor.data)
    
    def backward(self, grad):
        data = self.parents[0].data
        return np.maximum(np.sign(data), np.zeros_like(data)) * grad


class Sum(Function):
    type = ftype.reduce

    def forward(self, tensor):
        return np.sum(tensor.data)

    def backward(self, grad):
        return np.ones_like(self.parents[0].data) * grad
