"""
Core Tensor implementation with decentralized storage and advanced autograd support
"""

import numpy as np
from typing import Union, Tuple, Optional, List, Any
import uuid


class Tensor:
    """
    Advanced Omega Tensor with decentralized storage and next-gen autograd capabilities.
    
    Features:
    - Decentralized tensor storage with unique IDs
    - Automatic differentiation with computational graph tracking
    - Lazy evaluation support
    - Advanced memory management
    """
    
    _tensor_registry = {}  # Decentralized tensor storage
    
    def __init__(self, data, requires_grad=False, _children=(), _op='', dtype=None):
        """
        Initialize an Omega Tensor.
        
        Args:
            data: Input data (array-like or scalar)
            requires_grad: Whether to track gradients
            _children: Parent tensors in computational graph
            _op: Operation that created this tensor
            dtype: Data type for the tensor
        """
        # Convert data to numpy array
        if isinstance(data, Tensor):
            self.data = data.data.copy()
        else:
            self.data = np.array(data, dtype=dtype if dtype else np.float32)
        
        # Decentralized storage
        self.id = str(uuid.uuid4())
        Tensor._tensor_registry[self.id] = self
        
        # Autograd properties
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        
        # Advanced features
        self._lazy_eval = False
        self._distributed = False
        self._version = 0
        
    @property
    def shape(self):
        """Return tensor shape"""
        return self.data.shape
    
    @property
    def ndim(self):
        """Return number of dimensions"""
        return self.data.ndim
    
    @property
    def size(self):
        """Return total number of elements"""
        return self.data.size
    
    @property
    def dtype(self):
        """Return data type"""
        return self.data.dtype
    
    def __repr__(self):
        return f"Tensor(id={self.id[:8]}..., shape={self.shape}, grad_fn={self._op if self._op else None})"
    
    def __str__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
    
    # =====================
    # Arithmetic Operations
    # =====================
    
    def __add__(self, other):
        """Addition operation with autograd support"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, 
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='add')
        
        def _backward():
            if self.requires_grad:
                # Handle broadcasting in gradients
                grad = out.grad
                ndims_added = len(out.shape) - len(self.shape)
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = grad if self.grad is None else self.grad + grad
            
            if other.requires_grad:
                grad = out.grad
                ndims_added = len(out.shape) - len(other.shape)
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad = grad if other.grad is None else other.grad + grad
        
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        """Element-wise multiplication with autograd support"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='mul')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                ndims_added = len(out.shape) - len(self.shape)
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(self.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                self.grad = grad if self.grad is None else self.grad + grad
            
            if other.requires_grad:
                grad = out.grad * self.data
                ndims_added = len(out.shape) - len(other.shape)
                for _ in range(ndims_added):
                    grad = grad.sum(axis=0)
                for i, dim in enumerate(other.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                other.grad = grad if other.grad is None else other.grad + grad
        
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        """Power operation with autograd support"""
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Tensor(self.data ** other,
                    requires_grad=self.requires_grad,
                    _children=(self,), _op=f'pow{other}')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (other * self.data ** (other - 1))
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def __neg__(self):
        """Negation operation"""
        return self * -1
    
    def __sub__(self, other):
        """Subtraction operation"""
        return self + (-other if isinstance(other, Tensor) else -Tensor(other))
    
    def __truediv__(self, other):
        """Division operation"""
        return self * (other ** -1 if isinstance(other, Tensor) else Tensor(other) ** -1)
    
    def __radd__(self, other):
        """Reverse addition"""
        return self + other
    
    def __rmul__(self, other):
        """Reverse multiplication"""
        return self * other
    
    def __rsub__(self, other):
        """Reverse subtraction"""
        return Tensor(other) + (-self)
    
    def __rtruediv__(self, other):
        """Reverse division"""
        return Tensor(other) * (self ** -1)
    
    def __matmul__(self, other):
        """Matrix multiplication with autograd support"""
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,
                    requires_grad=self.requires_grad or other.requires_grad,
                    _children=(self, other), _op='matmul')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad @ other.data.swapaxes(-2, -1)
                self.grad = grad if self.grad is None else self.grad + grad
            
            if other.requires_grad:
                grad = self.data.swapaxes(-2, -1) @ out.grad
                other.grad = grad if other.grad is None else other.grad + grad
        
        out._backward = _backward
        return out
    
    # =====================
    # Advanced Operations
    # =====================
    
    def reshape(self, *shape):
        """Reshape tensor with gradient support"""
        out = Tensor(self.data.reshape(*shape),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='reshape')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad.reshape(self.shape)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def transpose(self, *axes):
        """Transpose tensor with gradient support"""
        out = Tensor(np.transpose(self.data, axes if axes else None),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='transpose')
        
        def _backward():
            if self.requires_grad:
                if axes:
                    # Invert the permutation
                    inv_axes = [0] * len(axes)
                    for i, ax in enumerate(axes):
                        inv_axes[ax] = i
                    grad = np.transpose(out.grad, inv_axes)
                else:
                    grad = np.transpose(out.grad)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False):
        """Sum reduction with gradient support"""
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='sum')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    if not keepdims:
                        grad = np.expand_dims(grad, axis=axis)
                    grad = np.broadcast_to(grad, self.shape)
                else:
                    grad = np.broadcast_to(grad, self.shape)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        """Mean reduction with gradient support"""
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='mean')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None:
                    if isinstance(axis, int):
                        numel = self.shape[axis]
                        if not keepdims:
                            grad = np.expand_dims(grad, axis=axis)
                    else:
                        # axis is a tuple or list
                        numel = np.prod([self.shape[a] for a in axis])
                        if not keepdims:
                            # Expand dims for each axis in sorted order
                            for ax in sorted(axis):
                                grad = np.expand_dims(grad, axis=ax)
                    grad = np.broadcast_to(grad, self.shape) / numel
                else:
                    grad = np.broadcast_to(grad, self.shape) / self.size
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def exp(self):
        """Exponential function with gradient support"""
        out = Tensor(np.exp(self.data),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='exp')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * out.data
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def log(self):
        """Natural logarithm with gradient support"""
        out = Tensor(np.log(self.data),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='log')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad / self.data
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def relu(self):
        """ReLU activation with gradient support"""
        out = Tensor(np.maximum(0, self.data),
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='relu')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (self.data > 0)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def sigmoid(self):
        """Sigmoid activation with gradient support"""
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig,
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='sigmoid')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * sig * (1 - sig)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    def tanh(self):
        """Tanh activation with gradient support"""
        t = np.tanh(self.data)
        out = Tensor(t,
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='tanh')
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (1 - t ** 2)
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
    
    # =====================
    # Autograd Methods
    # =====================
    
    def backward(self, gradient=None):
        """
        Compute gradients using reverse-mode automatic differentiation.
        
        This is the core of the next-gen autograd engine with topological sorting
        and efficient gradient accumulation.
        """
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise RuntimeError("gradient must be specified for non-scalar tensors")
        
        self.grad = gradient
        
        # Build computational graph in topological order
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # Apply chain rule in reverse order
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        """Reset gradients to None"""
        self.grad = None
    
    def detach(self):
        """Detach tensor from computational graph"""
        return Tensor(self.data.copy(), requires_grad=False)
    
    # =====================
    # Utility Methods
    # =====================
    
    def numpy(self):
        """Convert to numpy array"""
        return self.data.copy()
    
    def item(self):
        """Get scalar value"""
        return self.data.item()
    
    @staticmethod
    def zeros(*shape, requires_grad=False):
        """Create tensor filled with zeros"""
        return Tensor(np.zeros(shape), requires_grad=requires_grad)
    
    @staticmethod
    def ones(*shape, requires_grad=False):
        """Create tensor filled with ones"""
        return Tensor(np.ones(shape), requires_grad=requires_grad)
    
    @staticmethod
    def randn(*shape, requires_grad=False):
        """Create tensor with random normal values"""
        return Tensor(np.random.randn(*shape), requires_grad=requires_grad)
    
    @staticmethod
    def rand(*shape, requires_grad=False):
        """Create tensor with random uniform values"""
        return Tensor(np.random.rand(*shape), requires_grad=requires_grad)
    
    @staticmethod
    def eye(n, requires_grad=False):
        """Create identity matrix"""
        return Tensor(np.eye(n), requires_grad=requires_grad)
    
    def __getitem__(self, idx):
        """Indexing with gradient support"""
        out = Tensor(self.data[idx],
                    requires_grad=self.requires_grad,
                    _children=(self,), _op='getitem')
        
        def _backward():
            if self.requires_grad:
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad = grad if self.grad is None else self.grad + grad
        
        out._backward = _backward
        return out
