"""
Neural network module with advanced layers and utilities
"""

from .tensor import Tensor
import numpy as np


class Parameter(Tensor):
    """
    A trainable parameter (subclass of Tensor).
    """
    
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
        self.is_parameter = True


class Module:
    """
    Base class for all neural network modules.
    
    Your models should subclass this class.
    """
    
    def __init__(self):
        self._parameters = {}
        self._modules = {}
    
    def __call__(self, *args, **kwargs):
        """Make module callable"""
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Forward pass - must be implemented by subclasses"""
        raise NotImplementedError
    
    def parameters(self):
        """Return all parameters in the module"""
        params = []
        for param in self._parameters.values():
            params.append(param)
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def zero_grad(self):
        """Zero all parameter gradients"""
        for param in self.parameters():
            param.zero_grad()
    
    def train(self):
        """Set module to training mode"""
        self.training = True
        for module in self._modules.values():
            module.train()
    
    def eval(self):
        """Set module to evaluation mode"""
        self.training = False
        for module in self._modules.values():
            module.eval()
    
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)


class Linear(Module):
    """
    Fully connected linear layer.
    
    Applies: y = x @ W.T + b
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Xavier initialization
        k = np.sqrt(1.0 / in_features)
        self.weight = Parameter(
            Tensor.randn(out_features, in_features) * k
        )
        
        if bias:
            self.bias = Parameter(Tensor.zeros(out_features))
        else:
            self.bias = None
    
    def forward(self, x):
        """Forward pass"""
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    """ReLU activation function"""
    
    def forward(self, x):
        return x.relu()


class Sigmoid(Module):
    """Sigmoid activation function"""
    
    def forward(self, x):
        return x.sigmoid()


class Tanh(Module):
    """Tanh activation function"""
    
    def forward(self, x):
        return x.tanh()


class Dropout(Module):
    """
    Dropout layer for regularization.
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True
    
    def forward(self, x):
        if self.training and self.p > 0:
            # Create dropout mask
            mask = np.random.binomial(1, 1 - self.p, size=x.shape) / (1 - self.p)
            return x * Tensor(mask)
        return x


class BatchNorm1d(Module):
    """
    Batch normalization for 1D data.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.gamma = Parameter(Tensor.ones(num_features))
        self.beta = Parameter(Tensor.zeros(num_features))
        
        # Running statistics (not parameters)
        self.running_mean = Tensor.zeros(num_features)
        self.running_var = Tensor.ones(num_features)
        self.training = True
    
    def forward(self, x):
        if self.training:
            # Compute batch statistics
            mean = x.mean(axis=0)
            var = ((x - mean) ** 2).mean(axis=0)
            
            # Update running statistics (detached from graph)
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var.data
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        
        # Scale and shift
        out = self.gamma * x_norm + self.beta
        return out


class Embedding(Module):
    """
    Embedding layer for encoding tokens.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = Parameter(Tensor.randn(num_embeddings, embedding_dim))

    def forward(self, x):
        """
        Look up embeddings for indices x.
        x: Tensor of indices (int)
        """
        # Ensure x is integer type if it's a tensor
        if isinstance(x, Tensor):
            indices = x.data.astype(int)
        else:
            indices = np.array(x, dtype=int)

        return self.weight[indices]


class LayerNorm(Module):
    """
    Layer Normalization over the last dimension.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)
        self.eps = eps

        self.gamma = Parameter(Tensor.ones(*self.normalized_shape))
        self.beta = Parameter(Tensor.zeros(*self.normalized_shape))

    def forward(self, x):
        # Calculate mean and variance along the last dimension
        # Note: We assume normalization over the last dimension which is standard for Transformers
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)

        # Scale and shift
        return self.gamma * x_norm + self.beta


class Sequential(Module):
    """
    Sequential container for layers.
    """
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers
        for i, layer in enumerate(layers):
            self._modules[f'layer_{i}'] = layer
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MSELoss(Module):
    """Mean Squared Error loss"""
    
    def forward(self, pred, target):
        return ((pred - target) ** 2).mean()


class CrossEntropyLoss(Module):
    """Cross-entropy loss with softmax"""
    
    def forward(self, logits, targets):
        # Softmax
        exp_logits = logits.exp()
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        # Cross-entropy
        log_probs = probs.log()
        
        # Select target probabilities (simplified for demonstration)
        # In practice, would need proper indexing
        loss = -log_probs.mean()
        return loss


class Conv2d(Module):
    """
    2D Convolutional layer.
    
    Simplified implementation for demonstration.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        k = np.sqrt(1.0 / (in_channels * self.kernel_size[0] * self.kernel_size[1]))
        self.weight = Parameter(
            Tensor.randn(out_channels, in_channels, *self.kernel_size) * k
        )
        self.bias = Parameter(Tensor.zeros(out_channels))
    
    def forward(self, x):
        # Simplified conv2d - for demonstration
        # Full implementation would need im2col or other optimization
        batch_size, in_channels, height, width = x.shape
        kh, kw = self.kernel_size
        
        # Simple convolution (inefficient but correct)
        out_height = (height + 2 * self.padding - kh) // self.stride + 1
        out_width = (width + 2 * self.padding - kw) // self.stride + 1
        
        # For now, return a placeholder
        # Full implementation would require im2col
        out = Tensor.zeros(batch_size, self.out_channels, out_height, out_width)
        return out


class MaxPool2d(Module):
    """
    2D Max pooling layer.
    """
    
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride else kernel_size
    
    def forward(self, x):
        # Simplified max pooling
        # Full implementation would need proper indexing
        batch_size, channels, height, width = x.shape
        kh, kw = self.kernel_size
        
        out_height = (height - kh) // self.stride + 1
        out_width = (width - kw) // self.stride + 1
        
        out = Tensor.zeros(batch_size, channels, out_height, out_width)
        return out
