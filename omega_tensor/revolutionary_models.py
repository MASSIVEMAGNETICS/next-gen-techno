"""
Revolutionary Next-Gen Models for Omega Tensor
"""

import numpy as np
from .tensor import Tensor
from . import nn

class SpaceTimeEmbedding(nn.Module):
    """
    Space-Time Embedding Layer.
    Combines learnable embeddings with a unique Space-Time formula.
    """
    def __init__(self, num_embeddings, embedding_dim, max_len=5000):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.max_len = max_len
        self.embedding_dim = embedding_dim

        # Create constant positional encoding matrix ("Space-Time Grid")
        pe = np.zeros((max_len, embedding_dim), dtype=np.float32)
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, embedding_dim, 2) * -(np.log(10000.0) / embedding_dim))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = Tensor(pe, requires_grad=False)

        # Learnable time modulation factor
        self.time_factor = nn.Parameter(Tensor([1.0]))

    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.shape[1]
        embeddings = self.embedding(x)

        # Add positional encoding scaled by time factor
        # Slice PE to sequence length
        pe_slice = self.pe[:seq_len]

        # Reshape for broadcasting: (1, seq_len, dim)
        # Note: Tensor slicing returns a new Tensor.
        # We need to ensure shape is correct for broadcasting.
        # embeddings shape: (batch, seq_len, dim)
        # pe_slice shape: (seq_len, dim)
        # Broadcasting rules: (batch, seq_len, dim) + (seq_len, dim) works if batch matches or is 1.
        # Numpy broadcasting automatically prepends 1s.
        # So (seq_len, dim) broadcasts to (batch, seq_len, dim).

        # Modulation:
        pos_enc = pe_slice * self.time_factor

        return embeddings + pos_enc

class GravitronAttention(nn.Module):
    """
    Gravitron Attention Mechanism.
    Implements attention with a gravitational bias based on relative distance.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Gravitational constant (learnable strength of gravity)
        self.g_const = nn.Parameter(Tensor([1.0]))

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        # Reshape to (batch, seq, heads, head_dim) then transpose to (batch, heads, seq, head_dim)
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq) -> (batch, heads, seq, seq)
        scores = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # Gravitational bias ("Space-Time curvature")
        # Calculate distance matrix (seq_len, seq_len)
        indices = np.arange(seq_len)
        i_idx, j_idx = np.meshgrid(indices, indices, indexing='ij')
        dist_sq = (i_idx - j_idx) ** 2

        # Create gravity tensor: G / (1 + distance^2)
        gravity_dist = Tensor(dist_sq, requires_grad=False)
        gravity_bias = self.g_const / (1.0 + gravity_dist)

        # Reshape to broadcast: (1, 1, seq, seq) to match (batch, heads, seq, seq)
        gravity_bias = gravity_bias.reshape(1, 1, seq_len, seq_len)

        # Add gravitational bias to attention scores
        scores = scores + gravity_bias

        if mask is not None:
            scores = scores + mask

        # Softmax
        # For numerical stability: exp(x - max(x))
        # We compute max per row (last axis)
        # Tensor doesn't implement max(axis), so we use numpy on data and wrap in Tensor (detached)
        scores_max_data = np.max(scores.data, axis=-1, keepdims=True)
        scores_max = Tensor(scores_max_data, requires_grad=False)

        exp_scores = (scores - scores_max).exp()
        attn_weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        # Apply attention to values
        out = attn_weights @ v

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim) -> (batch, seq, embed_dim)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        return self.out_proj(out)

class NextGenTransformerBlock(nn.Module):
    """
    A single block of the Next Gen Transformer.
    """
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.attn = GravitronAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention sub-layer
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward sub-layer
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class NextGenTransformer(nn.Module):
    """
    The Next Generation Transformer with Space-Time Formula and Gravitron Attention.
    """
    def __init__(self, num_embeddings, embed_dim, num_heads, num_layers, feedforward_dim, num_classes):
        super().__init__()
        self.embedding = SpaceTimeEmbedding(num_embeddings, embed_dim)

        # Use Module list pattern manually since Sequential doesn't support complex forward args easily
        self.blocks = []
        for i in range(num_layers):
            block = NextGenTransformerBlock(embed_dim, num_heads, feedforward_dim)
            # Register block as submodule
            self._modules[f'block_{i}'] = block
            self.blocks.append(block)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask=None):
        # Embedding with Space-Time Formula
        x = self.embedding(x)

        # Transformer Blocks with Gravitron Attention
        for block in self.blocks:
            x = block(x, mask)

        # Classifier (simple pooling or last token)
        # Here we use mean pooling
        x_pool = x.mean(axis=1)

        return self.classifier(x_pool)
