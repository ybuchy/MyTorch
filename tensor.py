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
    @property
    def T(self) -> Tensor: return Transpose(self).apply()
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


class Transpose(Function):
    type = ftype.unary

    def forward(self, tensor):
        return tensor.data.T

    def backward(self, grad):
        return grad.T


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

    def forward(self, tensor, axis=None):
        self.tensor = tensor
        self.axis = axis
        self.m = np.amax(tensor.data, axis)
        return self.m

    # TODO (not readable at all, way too slow)
    def backward(self, grad):
        import time
        local_grad = np.zeros_like(self.tensor.data)
        if self.axis is None:
            s = 0
            it = np.nditer(self.tensor.data, flags=['multi_index'])
            for el in it:
                if el == self.m:
                    local_grad[it.multi_index] += 1
                    s += 1
            local_grad /= s
        else:
            a = time.monotonic()
            nptensor = np.swapaxes(self.tensor.data, self.axis, -1)
            indices = np.zeros_like(nptensor)
            # TODO IDEA: Is it faster to push do a element-wise translation to push the max to 1 and then set everything < 1 to 0? (still have branching but no (only less?) for loops)
            it = np.nditer(self.m, flags=['multi_index'])
            # redo _ and use it
            for _ in it:
                for ind, el in enumerate(nptensor[it.multi_index]):
                    if el == self.m[it.multi_index]:
                        indices[*it.multi_index, ind] += 1
            # useless to do zeros_like at the top
            b = time.monotonic()
            local_grad = np.swapaxes(indices, self.axis, -1)
            local_grad /= np.sum(local_grad, self.axis, keepdims=True)
            c = time.monotonic()
            grad = grad.reshape(*grad.shape, 1)
            axis_list = list(range(self.axis, len(grad.shape)))
            grad = np.moveaxis(grad, axis_list, axis_list[1:] + [self.axis])
            d = time.monotonic()
            print(f"{b-a:.5f}, {c-b:.5f}, {d-c:.5f}")
        return local_grad * grad


class Exp(Function):
    type = ftype.unary

    def forward(self, tensor):
        self.exp = np.exp(tensor.data)
        return self.exp

    def backward(self, grad):
        return self.exp * grad


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
