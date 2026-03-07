"""
Tests for Omega Tensor library
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omega_tensor import Tensor, nn, optim
import numpy as np


def test_tensor_creation():
    """Test tensor creation"""
    print("Testing tensor creation...")
    
    # From list
    t1 = Tensor([1, 2, 3])
    assert t1.shape == (3,)
    assert np.allclose(t1.data, [1, 2, 3])
    
    # From numpy array
    t2 = Tensor(np.array([[1, 2], [3, 4]]))
    assert t2.shape == (2, 2)
    
    # Static constructors
    t3 = Tensor.zeros(2, 3)
    assert t3.shape == (2, 3)
    assert np.allclose(t3.data, 0)
    
    t4 = Tensor.ones(3, 3)
    assert t4.shape == (3, 3)
    assert np.allclose(t4.data, 1)
    
    print("✓ Tensor creation tests passed")


def test_basic_operations():
    """Test basic arithmetic operations"""
    print("Testing basic operations...")
    
    x = Tensor([1, 2, 3])
    y = Tensor([4, 5, 6])
    
    # Addition
    z = x + y
    assert np.allclose(z.data, [5, 7, 9])
    
    # Subtraction
    z = x - y
    assert np.allclose(z.data, [-3, -3, -3])
    
    # Multiplication
    z = x * y
    assert np.allclose(z.data, [4, 10, 18])
    
    # Division
    z = x / Tensor([2, 2, 2])
    assert np.allclose(z.data, [0.5, 1, 1.5])
    
    # Power
    z = x ** 2
    assert np.allclose(z.data, [1, 4, 9])
    
    print("✓ Basic operations tests passed")


def test_matrix_multiplication():
    """Test matrix multiplication"""
    print("Testing matrix multiplication...")
    
    A = Tensor([[1, 2], [3, 4]])
    B = Tensor([[5, 6], [7, 8]])
    
    C = A @ B
    expected = np.array([[19, 22], [43, 50]])
    
    assert np.allclose(C.data, expected)
    
    print("✓ Matrix multiplication tests passed")


def test_gradients():
    """Test gradient computation"""
    print("Testing gradients...")
    
    # Simple gradient
    x = Tensor([2.0], requires_grad=True)
    y = x ** 2
    y.backward()
    
    # dy/dx = 2x = 4
    assert np.allclose(x.grad, [4.0])
    
    # Chain rule
    x = Tensor([3.0], requires_grad=True)
    y = Tensor([2.0], requires_grad=True)
    z = x * y
    w = z + x
    w.backward()
    
    # dw/dx = y + 1 = 3
    assert np.allclose(x.grad, [3.0])
    # dw/dy = x = 3
    assert np.allclose(y.grad, [3.0])
    
    print("✓ Gradient tests passed")


def test_activation_functions():
    """Test activation functions and their gradients"""
    print("Testing activation functions...")
    
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    
    # ReLU
    y = x.relu()
    assert np.allclose(y.data, [0.0, 0.0, 1.0])
    
    # Sigmoid
    x2 = Tensor([0.0], requires_grad=True)
    y = x2.sigmoid()
    assert np.allclose(y.data, [0.5], atol=1e-6)
    
    # Tanh
    x3 = Tensor([0.0], requires_grad=True)
    y = x3.tanh()
    assert np.allclose(y.data, [0.0], atol=1e-6)
    
    print("✓ Activation function tests passed")


def test_reshape_transpose():
    """Test reshape and transpose operations"""
    print("Testing reshape and transpose...")
    
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    
    # Reshape
    y = x.reshape(3, 2)
    assert y.shape == (3, 2)
    
    # Transpose
    y = x.transpose()
    assert y.shape == (3, 2)
    assert np.allclose(y.data, [[1, 4], [2, 5], [3, 6]])
    
    print("✓ Reshape and transpose tests passed")


def test_reduction_operations():
    """Test reduction operations"""
    print("Testing reduction operations...")
    
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    
    # Sum
    y = x.sum()
    assert np.allclose(y.data, 21)
    
    # Sum with axis
    y = x.sum(axis=0)
    assert np.allclose(y.data, [5, 7, 9])
    
    # Mean
    y = x.mean()
    assert np.allclose(y.data, 3.5)
    
    print("✓ Reduction operation tests passed")


def test_broadcasting():
    """Test broadcasting in operations"""
    print("Testing broadcasting...")
    
    x = Tensor([[1, 2, 3]], requires_grad=True)  # (1, 3)
    y = Tensor([10], requires_grad=True)  # (1,)
    
    z = x + y
    assert z.shape == (1, 3)
    assert np.allclose(z.data, [[11, 12, 13]])
    
    print("✓ Broadcasting tests passed")


def test_linear_layer():
    """Test linear layer"""
    print("Testing linear layer...")
    
    layer = nn.Linear(5, 3)
    x = Tensor.randn(2, 5)
    
    y = layer(x)
    assert y.shape == (2, 3)
    
    # Test backward
    loss = y.sum()
    loss.backward()
    
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    
    print("✓ Linear layer tests passed")


def test_sequential_model():
    """Test sequential model"""
    print("Testing sequential model...")
    
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    x = Tensor.randn(3, 10)
    y = model(x)
    
    assert y.shape == (3, 5)
    
    print("✓ Sequential model tests passed")


def test_optimizers():
    """Test optimizer functionality"""
    print("Testing optimizers...")
    
    # Create a simple parameter
    x = Tensor([5.0], requires_grad=True)
    
    # Test SGD
    optimizer = optim.SGD([x], lr=0.1)
    
    loss = (x - 2) ** 2
    optimizer.zero_grad()
    loss.backward()
    
    old_value = x.data.copy()
    optimizer.step()
    
    # Check that parameter was updated
    assert not np.allclose(x.data, old_value)
    
    # Test Adam
    x = Tensor([5.0], requires_grad=True)
    optimizer = optim.Adam([x], lr=0.1)
    
    loss = (x - 2) ** 2
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("✓ Optimizer tests passed")


def test_decentralized_storage():
    """Test decentralized tensor storage"""
    print("Testing decentralized storage...")
    
    initial_count = len(Tensor._tensor_registry)
    
    t1 = Tensor([1, 2, 3])
    t2 = Tensor([4, 5, 6])
    
    # Check that tensors are in registry
    assert t1.id in Tensor._tensor_registry
    assert t2.id in Tensor._tensor_registry
    
    # Check that registry grew
    assert len(Tensor._tensor_registry) == initial_count + 2
    
    print("✓ Decentralized storage tests passed")


def test_next_gen_models():
    """Test revolutionary next-gen models"""
    print("Testing next-gen models...")

    from omega_tensor.revolutionary_models import NextGenTransformer, SpaceTimeEmbedding

    # Test SpaceTimeEmbedding
    emb = SpaceTimeEmbedding(10, 4, max_len=10)
    # Using numpy array directly for indices as now supported/handled
    x = Tensor(np.array([[0, 1, 2]]))
    y = emb(x)
    assert y.shape == (1, 3, 4)
    y.sum().backward()
    assert emb.embedding.weight.grad is not None
    assert emb.time_factor.grad is not None

    # Test NextGenTransformer
    model = NextGenTransformer(
        num_embeddings=20,
        embed_dim=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        num_classes=2
    )

    x = Tensor(np.zeros((2, 5))) # Batch 2, Seq 5
    out = model(x)
    assert out.shape == (2, 2)

    loss = out.sum()
    loss.backward()

    assert model.classifier.weight.grad is not None

    print("✓ Next-gen models tests passed")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Running Omega Tensor Tests")
    print("=" * 60)
    print()
    
    test_tensor_creation()
    test_basic_operations()
    test_matrix_multiplication()
    test_gradients()
    test_activation_functions()
    test_reshape_transpose()
    test_reduction_operations()
    test_broadcasting()
    test_linear_layer()
    test_sequential_model()
    test_optimizers()
    test_decentralized_storage()
    test_next_gen_models()
    
    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
