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

        # only if grad_tracked, but how to differentiate between leaf and rest?
        if requires_grad:
            self.grad = np.zeros(self.data.shape)


    # TODO
    @property
    def grad_tracked(self):
        pass

    def sum(self) -> Tensor:
        # This will be for grad to work (why needed?)
        pass

    def backward(self, grad: Optional[bool]=None) -> None:
        if self.back_fn is None: raise AttributeError("back function missing")
        if not self.back_fn.requires_grad: return

        for parent in back_fn.parents:
            if not parent.requires_grad: continue
            # example: linear, then the units are one tensor and weights are another tensor. optimizer will track weight tensor so this will add grad to weights but only forward the gradient through the layer
            if parent.grad_tracked:
                parent.grad += back_fn.backward(grad)
            parent.backward(grad)
        pass
        
    def dot(self, tensor: Tensor) -> Tensor: return self.__matmul__(tensor)
        new_data = self.data @ rhs.data
        requires_grad = self.requires_grad or rhs.requires_grad
        # TODO change to use Function (Dot) forward
        back_fn = Dot(self, rhs)
        return Tensor(new_data, requires_grad=requires_grad, back_fn=back_fn)

    def __add__(self, rhs: Tensor) -> Tensor:
        requires_grad = self.requires_grad or rhs.requires_grad
        new_data = self.data + rhs.data
        return Tensor(new_data, requires_grad=True)#TODO, back_fn=...

    def __matmul__(self, rhs: Tensor) -> Tensor: return self.dot(rhs)

    def __str__(self): return f"Tensor({np.array2string(self.data)})"

class Function:
    # TODO
    def __init__(self, *tensors: Sequence[Tensor]):
        self.parents = tensors
        self.requires_grad = any(tensor.requires_grad for tensor in tensors)

    def forward(self):
        raise NotImplementedError(f"forward function of {type(self)} not implemented")

    def backward(self):
        raise NotImplementedError(f"backward function of {type(self)} not implemented")

    # TODO
    def apply(self):
        pass
