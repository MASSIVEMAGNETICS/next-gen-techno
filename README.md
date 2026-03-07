# Omega Tensor

**Advanced Decentralized Tensor Library with Next-Gen Autograd Engine**

Omega Tensor is a custom-built tensor computation library featuring a revolutionary autograd engine, decentralized tensor storage, and post-autograd optimizations for high-performance deep learning.

## 🌟 Key Features

### 1. **Decentralized Tensor Storage**
- Unique ID-based tensor registry for distributed computation
- Efficient memory management with version tracking
- Support for distributed tensor operations across nodes

### 2. **Next-Gen Autograd Engine**
- Automatic differentiation with dynamic computational graph
- Topological sorting for efficient gradient computation
- Support for complex gradient flows and broadcasting
- Custom Function API for extending with new differentiable operations

### 3. **Revolutionary Post-Autograd Features**
- **Gradient Checkpointing**: Trade computation for memory by recomputing activations during backward pass
- **Lazy Evaluation**: Delay computation and automatically fuse operations for efficiency
- **Distributed Autograd**: Coordinate gradient computation across distributed nodes

### 4. **Comprehensive Neural Network API**
- Modular `nn.Module` system similar to PyTorch
- Common layers: Linear, Conv2d, BatchNorm, Dropout, etc.
- Activation functions: ReLU, Sigmoid, Tanh with automatic gradient support
- Loss functions: MSE, CrossEntropy

### 5. **Advanced Optimizers**
- SGD with momentum
- Adam (Adaptive Moment Estimation)
- AdamW (Adam with decoupled weight decay)
- RMSprop

## 🚀 Quick Start

### Installation

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

### Basic Usage

```python
from omega_tensor import Tensor

# Create tensors
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
y = Tensor([4.0, 5.0, 6.0], requires_grad=True)

# Perform operations
z = x + y
w = z * 2
loss = w.sum()

# Compute gradients automatically
loss.backward()

print(f"x.grad = {x.grad}")  # Gradients computed!
```

### Building a Neural Network

```python
from omega_tensor import Tensor, nn, optim

# Define a neural network
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model(X)
    loss = ((predictions - y) ** 2).mean()
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
```

## 📚 Core Concepts

### Tensor Operations

Omega Tensor supports a wide range of operations with automatic gradient computation:

**Arithmetic Operations:**
- Addition, subtraction, multiplication, division
- Power, negation
- Broadcasting support

**Matrix Operations:**
- Matrix multiplication (`@`)
- Transpose, reshape

**Reduction Operations:**
- Sum, mean (with axis support)

**Activation Functions:**
- ReLU, Sigmoid, Tanh
- Exponential, logarithm

### Autograd Engine

The autograd engine automatically tracks operations and computes gradients:

```python
x = Tensor([2.0], requires_grad=True)
y = x ** 2  # y = 4
y.backward()
print(x.grad)  # dy/dx = 2x = 4.0
```

The engine:
1. Builds a computational graph during forward pass
2. Uses topological sorting for efficient traversal
3. Applies the chain rule in reverse order
4. Handles broadcasting and shape changes correctly

### Decentralized Storage

Each tensor gets a unique UUID and is registered in a decentralized storage system:

```python
t = Tensor([1, 2, 3])
print(t.id)  # Unique identifier
print(Tensor._tensor_registry[t.id])  # Access from registry
```

This enables:
- Distributed computation across nodes
- Efficient tensor lookup and sharing
- Version tracking for tensor updates

## 🔬 Advanced Features

### Gradient Checkpointing

Save memory by recomputing activations during backward pass:

```python
from omega_tensor.autograd import checkpoint

def expensive_function(x):
    return x.exp().tanh()

# Only stores input, recomputes during backward
output = checkpoint(expensive_function, x)
```

### Lazy Evaluation

Operations are automatically fused for efficiency:

```python
from omega_tensor.autograd import LazyEvaluation

lazy = LazyEvaluation()
# Operations are queued and fused
lazy.add_operation('add', x, y)
lazy.add_operation('mul', result, 2)
lazy.evaluate()  # Executes as single fused kernel
```

### Distributed Autograd

Coordinate gradient computation across distributed nodes:

```python
from omega_tensor.autograd import enable_distributed

x = Tensor([1, 2, 3], requires_grad=True)
x = enable_distributed(x)
# Gradient computation is now distributed
```

## 📖 Examples

See `examples.py` for comprehensive examples including:
- Basic operations and autograd
- Matrix multiplication
- Neural network training
- Activation functions
- Decentralized storage
- Computational graphs
- Broadcasting
- Optimizer comparison

Run examples:
```bash
python examples.py
```

## 🧪 Testing

Run the test suite:

```bash
python tests.py
```

Or with pytest:
```bash
pip install pytest
pytest tests.py -v
```

## 🏗️ Architecture

### Core Components

```
omega_tensor/
├── tensor.py       # Core Tensor class with operations
├── autograd.py     # Autograd engine and advanced features
├── nn.py           # Neural network modules
└── optim.py        # Optimization algorithms
```

### Tensor Class

The `Tensor` class is the fundamental building block:
- Wraps numpy arrays for computation
- Tracks computational graph for autograd
- Unique ID for decentralized storage
- Lazy evaluation support

### Autograd Engine

The autograd engine provides:
- Dynamic computational graph construction
- Reverse-mode automatic differentiation
- Custom backward functions for each operation
- Efficient topological sorting

## 🎯 Design Philosophy

1. **Simplicity**: Clean, readable code that's easy to understand and extend
2. **Modularity**: Separate concerns with clear interfaces
3. **Efficiency**: Optimized operations with numpy backend
4. **Flexibility**: Easy to add custom operations and layers
5. **Innovation**: Revolutionary features like gradient checkpointing and lazy evaluation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional optimizers (LAMB, RAdam, etc.)
- More neural network layers
- GPU support with CuPy
- Distributed training features
- Performance optimizations
- Documentation improvements

## 📄 License

MIT License - feel free to use in your projects!

## 🙏 Acknowledgments

Inspired by PyTorch, TensorFlow, and JAX, but built from scratch with revolutionary features for next-generation deep learning.

---

**Built with ❤️ by MASSIVEMAGNETICS**
