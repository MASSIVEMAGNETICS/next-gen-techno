"""
Omega Tensor: Advanced Decentralized Tensor Library with Next-Gen Autograd Engine
"""

from .tensor import Tensor
from .autograd import Function, no_grad
from .nn import Module, Parameter
from . import nn
from . import optim

__version__ = "0.1.0"
__all__ = ["Tensor", "Function", "no_grad", "Module", "Parameter", "nn", "optim"]
