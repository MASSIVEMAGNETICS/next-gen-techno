"""
Examples demonstrating the Omega Tensor library capabilities
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omega_tensor import Tensor, nn, optim
import numpy as np


def example_basic_operations():
    """Example 1: Basic tensor operations and autograd"""
    print("=" * 60)
    print("Example 1: Basic Tensor Operations with Autograd")
    print("=" * 60)
    
    # Create tensors
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    
    # Perform operations
    z = x + y
    w = z * 2
    loss = w.sum()
    
    print(f"x = {x.data}")
    print(f"y = {y.data}")
    print(f"z = x + y = {z.data}")
    print(f"w = z * 2 = {w.data}")
    print(f"loss = w.sum() = {loss.data}")
    
    # Backward pass
    loss.backward()
    
    print(f"\nGradients:")
    print(f"x.grad = {x.grad}")
    print(f"y.grad = {y.grad}")
    print()


def example_matrix_operations():
    """Example 2: Matrix multiplication and advanced operations"""
    print("=" * 60)
    print("Example 2: Matrix Operations")
    print("=" * 60)
    
    # Create matrices
    A = Tensor([[1, 2], [3, 4]], requires_grad=True)
    B = Tensor([[5, 6], [7, 8]], requires_grad=True)
    
    # Matrix multiplication
    C = A @ B
    
    print(f"A =\n{A.data}")
    print(f"B =\n{B.data}")
    print(f"C = A @ B =\n{C.data}")
    
    # Compute loss and gradients
    loss = C.sum()
    loss.backward()
    
    print(f"\nGradients:")
    print(f"A.grad =\n{A.grad}")
    print(f"B.grad =\n{B.grad}")
    print()


def example_neural_network():
    """Example 3: Simple neural network"""
    print("=" * 60)
    print("Example 3: Neural Network Training")
    print("=" * 60)
    
    # Create a simple 2-layer network
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Generate synthetic data
    np.random.seed(42)
    X_train = Tensor(np.random.randn(100, 10))
    y_train = Tensor(np.random.randn(100, 1))
    
    # Training loop
    print("Training neural network...")
    for epoch in range(5):
        # Forward pass
        predictions = model(X_train)
        
        # Compute loss
        loss = ((predictions - y_train) ** 2).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        print(f"Epoch {epoch + 1}/5, Loss: {loss.item():.6f}")
    
    print()


def example_activation_functions():
    """Example 4: Activation functions with gradients"""
    print("=" * 60)
    print("Example 4: Activation Functions")
    print("=" * 60)
    
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    
    # ReLU
    relu_out = x.relu()
    print(f"Input: {x.data}")
    print(f"ReLU: {relu_out.data}")
    
    # Sigmoid
    sigmoid_out = x.sigmoid()
    print(f"Sigmoid: {sigmoid_out.data}")
    
    # Tanh
    tanh_out = x.tanh()
    print(f"Tanh: {tanh_out.data}")
    
    # Compute gradients
    loss = relu_out.sum()
    loss.backward()
    print(f"\nReLU gradient: {x.grad}")
    print()


def example_decentralized_storage():
    """Example 5: Decentralized tensor storage"""
    print("=" * 60)
    print("Example 5: Decentralized Tensor Storage")
    print("=" * 60)
    
    # Create tensors - each gets a unique ID
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    t3 = Tensor([7, 8, 9])
    
    print(f"Tensor 1: ID={t1.id[:8]}..., data={t1.data}")
    print(f"Tensor 2: ID={t2.id[:8]}..., data={t2.data}")
    print(f"Tensor 3: ID={t3.id[:8]}..., data={t3.data}")
    
    # Access from registry
    print(f"\nTotal tensors in registry: {len(Tensor._tensor_registry)}")
    print()


def example_computational_graph():
    """Example 6: Computational graph visualization"""
    print("=" * 60)
    print("Example 6: Computational Graph")
    print("=" * 60)
    
    # Build a computation
    x = Tensor([2.0], requires_grad=True)
    y = Tensor([3.0], requires_grad=True)
    
    # Complex computation
    a = x * y  # mul
    b = a + x  # add
    c = b ** 2  # pow
    d = c.exp()  # exp
    
    print("Computation graph:")
    print(f"x = {x.data[0]:.2f}")
    print(f"y = {y.data[0]:.2f}")
    print(f"a = x * y = {a.data[0]:.2f} (op: {a._op})")
    print(f"b = a + x = {b.data[0]:.2f} (op: {b._op})")
    print(f"c = b ** 2 = {c.data[0]:.2f} (op: {c._op})")
    print(f"d = c.exp() = {d.data[0]:.2f} (op: {d._op})")
    
    # Backpropagate
    d.backward()
    
    print(f"\nGradients:")
    print(f"∂d/∂x = {x.grad[0]:.6f}")
    print(f"∂d/∂y = {y.grad[0]:.6f}")
    print()


def example_broadcasting():
    """Example 7: Broadcasting in operations"""
    print("=" * 60)
    print("Example 7: Broadcasting")
    print("=" * 60)
    
    # Create tensors of different shapes
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)  # (2, 3)
    y = Tensor([10, 20, 30], requires_grad=True)  # (3,)
    
    # Broadcasting addition
    z = x + y
    
    print(f"x.shape = {x.shape}, data =\n{x.data}")
    print(f"y.shape = {y.shape}, data = {y.data}")
    print(f"z = x + y, shape = {z.shape}, data =\n{z.data}")
    
    # Backward pass handles broadcasting correctly
    loss = z.sum()
    loss.backward()
    
    print(f"\nGradients:")
    print(f"x.grad.shape = {x.grad.shape}, data =\n{x.grad}")
    print(f"y.grad.shape = {y.grad.shape}, data = {y.grad}")
    print()


def example_advanced_optimizers():
    """Example 8: Different optimizers comparison"""
    print("=" * 60)
    print("Example 8: Optimizer Comparison")
    print("=" * 60)
    
    # Simple optimization problem: minimize (x - 5)^2
    def train_with_optimizer(optimizer_class, name, **kwargs):
        x = Tensor([0.0], requires_grad=True)
        target = 5.0
        
        optimizer = optimizer_class([x], **kwargs)
        
        print(f"\n{name}:")
        for i in range(10):
            loss = (x - target) ** 2
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 2 == 0:
                print(f"  Step {i}: x = {x.data[0]:.6f}, loss = {loss.data[0]:.6f}")
        
        return x.data[0]
    
    # Test different optimizers
    train_with_optimizer(optim.SGD, "SGD", lr=0.1)
    train_with_optimizer(optim.Adam, "Adam", lr=0.5)
    train_with_optimizer(optim.RMSprop, "RMSprop", lr=0.5)
    print()


def run_all_examples():
    """Run all examples"""
    example_basic_operations()
    example_matrix_operations()
    example_neural_network()
    example_activation_functions()
    example_decentralized_storage()
    example_computational_graph()
    example_broadcasting()
    example_advanced_optimizers()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_examples()
