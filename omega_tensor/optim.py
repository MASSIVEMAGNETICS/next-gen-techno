"""
Optimization algorithms for training neural networks
"""

import numpy as np


class Optimizer:
    """Base class for all optimizers"""
    
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr
    
    def zero_grad(self):
        """Zero all parameter gradients"""
        for param in self.parameters:
            param.zero_grad()
    
    def step(self):
        """Perform optimization step - must be implemented by subclasses"""
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Args:
        parameters: Iterable of parameters to optimize
        lr: Learning rate
        momentum: Momentum factor (default: 0)
        weight_decay: Weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, parameters, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        """Perform a single optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Add momentum
            if self.momentum != 0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                grad = self.velocity[i]
            
            # Update parameters
            param.data -= self.lr * grad


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation).
    
    Args:
        parameters: Iterable of parameters to optimize
        lr: Learning rate
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize moments
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]
        self.t = 0
    
    def step(self):
        """Perform a single optimization step"""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update biased first moment estimate
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            
            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop(Optimizer):
    """
    RMSprop optimizer.
    
    Args:
        parameters: Iterable of parameters to optimize
        lr: Learning rate
        alpha: Smoothing constant (default: 0.99)
        eps: Term for numerical stability (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, parameters, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        
        # Initialize running average
        self.v = [np.zeros_like(p.data) for p in self.parameters]
    
    def step(self):
        """Perform a single optimization step"""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Add weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Update running average
            self.v[i] = self.alpha * self.v[i] + (1 - self.alpha) * (grad ** 2)
            
            # Update parameters
            param.data -= self.lr * grad / (np.sqrt(self.v[i]) + self.eps)


class AdamW(Adam):
    """
    AdamW optimizer (Adam with decoupled weight decay).
    
    This is a more principled version of weight decay for Adam.
    """
    
    def step(self):
        """Perform a single optimization step"""
        self.t += 1
        
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # Update biased first moment estimate
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
            
            # Update parameters with decoupled weight decay
            param.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * param.data)
