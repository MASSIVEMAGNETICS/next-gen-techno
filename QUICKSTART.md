# Omega Tensor - Quick Start Guide

## What is Omega Tensor?

Omega Tensor is a revolutionary tensor computation library built from scratch with:
- **Decentralized tensor storage** with unique IDs for distributed computation
- **Next-gen autograd engine** with dynamic computational graphs
- **Post-autograd optimizations** including gradient checkpointing and lazy evaluation

## Installation

```bash
git clone https://github.com/MASSIVEMAGNETICS/omega-tensor.git
cd omega-tensor
pip install -e .
```

## 5-Minute Tutorial

### 1. Basic Tensor Operations

```python
from omega_tensor import Tensor

# Create tensors
x = Tensor([1, 2, 3], requires_grad=True)
y = Tensor([4, 5, 6], requires_grad=True)

# Perform operations
z = x + y
print(z.data)  # [5, 7, 9]

# Automatic differentiation
loss = z.sum()
loss.backward()
print(x.grad)  # [1, 1, 1]
```

### 2. Matrix Operations

```python
A = Tensor([[1, 2], [3, 4]], requires_grad=True)
B = Tensor([[5, 6], [7, 8]], requires_grad=True)

# Matrix multiplication
C = A @ B
print(C.data)  # [[19, 22], [43, 50]]
```

### 3. Neural Networks

```python
from omega_tensor import nn, optim
import numpy as np

# Define model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
X = Tensor(np.random.randn(100, 10))
y = Tensor(np.random.randn(100, 1))

for epoch in range(10):
    pred = model(X)
    loss = ((pred - y) ** 2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Loss: {loss.item():.4f}")
```

### 4. Advanced Features

#### Gradient Checkpointing (save memory)
```python
from omega_tensor.autograd import checkpoint

def expensive_layer(x):
    return x.exp().tanh().relu()

# Memory-efficient computation
output = checkpoint(expensive_layer, input_tensor)
```

#### Decentralized Storage
```python
# Each tensor has a unique ID
t = Tensor([1, 2, 3])
print(t.id)  # UUID

# Access from global registry
Tensor._tensor_registry[t.id]
```

## Key Features

### Supported Operations
- **Arithmetic**: +, -, *, /, **
- **Matrix**: @, transpose, reshape
- **Reduction**: sum, mean
- **Activation**: relu, sigmoid, tanh, exp, log
- **Broadcasting**: Automatic shape alignment

### Optimizers
- SGD with momentum
- Adam
- AdamW
- RMSprop

### Neural Network Layers
- Linear (fully connected)
- ReLU, Sigmoid, Tanh
- Dropout
- BatchNorm1d
- Conv2d (basic)
- MaxPool2d (basic)

## Examples

Run comprehensive examples:
```bash
python examples.py
```

Run tests:
```bash
python tests.py
```

## Revolutionary Features

### 1. Decentralized Tensor Storage
Every tensor gets a unique UUID and is stored in a decentralized registry, enabling:
- Distributed computation across nodes
- Efficient tensor sharing and lookup
- Version tracking

### 2. Next-Gen Autograd Engine
- Dynamic computational graph construction
- Topological sorting for efficient gradient computation
- Correct handling of broadcasting and complex shapes
- Custom backward functions

### 3. Post-Autograd Optimizations
- **Gradient Checkpointing**: Trade computation for memory
- **Lazy Evaluation**: Automatic operation fusion
- **Distributed Autograd**: Coordinate gradients across nodes

## Learn More

- Full documentation: See README.md
- API reference: Check docstrings in source code
- Examples: See examples.py
- Tests: See tests.py

## Philosophy

Built from scratch to demonstrate:
1. How modern deep learning frameworks work internally
2. Clean, readable implementation of complex concepts
3. Revolutionary optimizations for efficiency

## Contributing

We welcome contributions! Areas for improvement:
- GPU support
- More optimizers and layers
- Performance optimizations
- Distributed training features

---

**Happy Tensoring! 🚀**
