import numpy as np
from omega_tensor.tensor import Tensor
from omega_tensor.revolutionary_models import NextGenTransformer
import time

def test_next_gen_transformer():
    print("Initializing NextGenTransformer...")
    model = NextGenTransformer(
        num_embeddings=1000,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        num_classes=10
    )

    # Create dummy input
    batch_size = 8
    seq_len = 32
    x = Tensor(np.random.randint(0, 1000, size=(batch_size, seq_len)))

    print(f"Input shape: {x.shape}")

    # Forward pass
    start_time = time.time()
    output = model(x)
    end_time = time.time()

    print(f"Forward pass completed in {end_time - start_time:.4f}s")
    print(f"Output shape: {output.shape}")

    assert output.shape == (batch_size, 10)

    # Compute loss
    target = Tensor(np.random.randn(batch_size, 10)) # Dummy regression targets for simplicity
    loss = ((output - target) ** 2).mean()

    print(f"Loss: {loss.item()}")

    # Backward pass
    print("Starting backward pass...")
    start_time = time.time()
    loss.backward()
    end_time = time.time()
    print(f"Backward pass completed in {end_time - start_time:.4f}s")

    # Verify gradients
    print("Verifying gradients...")
    assert model.classifier.weight.grad is not None
    print("Classifier gradients computed.")
    assert model.embedding.time_factor.grad is not None
    print(f"Time factor grad: {model.embedding.time_factor.grad}")
    assert model.blocks[0].attn.g_const.grad is not None
    print(f"Gravitron constant grad: {model.blocks[0].attn.g_const.grad}")

    print("All tests passed!")

if __name__ == "__main__":
    test_next_gen_transformer()
