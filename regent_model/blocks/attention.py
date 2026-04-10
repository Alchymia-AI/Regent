"""
Grouped Query Attention (GQA) block for hybrid Mamba-2 architecture.

These layers appear every ~8 Mamba layers to provide precise recall
over recent context — compensating for SSM's lossy state compression.
Uses sliding window attention to maintain Mamba's memory efficiency.
"""

import math

import mlx.core as mx
import mlx.nn as nn


class GQABlock(nn.Module):
    """
    Grouped Query Attention with sliding window.

    GQA uses fewer KV heads than query heads, reducing KV cache memory
    while retaining most of full multi-head attention quality.
    """

    def __init__(
        self,
        d_model: int,
        n_q_heads: int,
        n_kv_heads: int,
        head_dim: int,
        window_size: int = 1024,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.n_groups = n_q_heads // n_kv_heads
        self.scale = head_dim**-0.5

        assert n_q_heads % n_kv_heads == 0, (
            f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        )

        self.q_proj = nn.Linear(d_model, n_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, d_model, bias=False)

        self.norm = nn.RMSNorm(d_model, eps=norm_eps)

    def _apply_rotary_pos_emb(
        self, x: mx.array, offset: int = 0
    ) -> mx.array:
        """
        Apply RoPE (Rotary Position Embedding) to queries/keys.

        Args:
            x: (batch, seq_len, n_heads, head_dim)
            offset: position offset for cached inference
        """
        _, seq_len, _, head_dim = x.shape

        # Compute rotation frequencies
        inv_freq = 1.0 / (10000.0 ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim))
        positions = mx.arange(offset, offset + seq_len).astype(mx.float32)
        freqs = positions[:, None] * inv_freq[None, :]  # (seq_len, head_dim/2)

        cos = mx.cos(freqs)  # (seq_len, head_dim/2)
        sin = mx.sin(freqs)  # (seq_len, head_dim/2)

        # Split x into pairs and rotate
        x1 = x[:, :, :, 0::2]  # even indices
        x2 = x[:, :, :, 1::2]  # odd indices

        # Apply rotation
        y1 = x1 * cos[None, :, None, :] - x2 * sin[None, :, None, :]
        y2 = x1 * sin[None, :, None, :] + x2 * cos[None, :, None, :]

        # Interleave back
        # Stack along last dim and reshape
        y = mx.stack([y1, y2], axis=-1).reshape(x.shape)

        return y

    def __call__(
        self,
        x: mx.array,
        cache: dict | None = None,
    ) -> tuple[mx.array, dict | None]:
        """
        Forward pass with optional KV cache for inference.

        Args:
            x: (batch, seq_len, d_model)
            cache: optional dict with 'k' and 'v' tensors from previous steps

        Returns:
            output: (batch, seq_len, d_model)
            new_cache: updated KV cache
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).reshape(batch, seq_len, self.n_q_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Determine position offset from cache
        offset = 0
        if cache is not None and "k" in cache:
            offset = cache["k"].shape[1]

        # Apply RoPE to Q and K
        q = self._apply_rotary_pos_emb(q, offset=offset)
        k = self._apply_rotary_pos_emb(k, offset=offset)

        # Append to KV cache
        if cache is not None and "k" in cache:
            k = mx.concatenate([cache["k"], k], axis=1)
            v = mx.concatenate([cache["v"], v], axis=1)

            # Sliding window: trim to window_size
            if k.shape[1] > self.window_size:
                k = k[:, -self.window_size :]
                v = v[:, -self.window_size :]

        new_cache = {"k": k, "v": v}

        kv_len = k.shape[1]

        # Expand KV heads to match Q heads (grouped query attention)
        # k: (batch, kv_len, n_kv_heads, head_dim) -> (batch, kv_len, n_q_heads, head_dim)
        if self.n_groups > 1:
            k = mx.repeat(k, self.n_groups, axis=2)
            v = mx.repeat(v, self.n_groups, axis=2)

        # Transpose to (batch, n_heads, seq_len, head_dim) for attention
        q = q.transpose(0, 2, 1, 3)  # (batch, n_q_heads, seq_len, head_dim)
        k = k.transpose(0, 2, 1, 3)  # (batch, n_q_heads, kv_len, head_dim)
        v = v.transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (batch, n_q_heads, seq_len, kv_len)

        # Causal mask: each query can only attend to positions <= its own
        # With sliding window, positions are already trimmed
        causal_mask = mx.tri(seq_len, kv_len, k=kv_len - seq_len)  # (seq_len, kv_len)
        attn_weights = mx.where(
            causal_mask[None, None, :, :],
            attn_weights,
            mx.array(float("-inf")),
        )

        attn_weights = mx.softmax(attn_weights, axis=-1)

        # Apply attention to values
        attn_output = attn_weights @ v  # (batch, n_q_heads, seq_len, head_dim)

        # Transpose back and project
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (batch, seq_len, n_q_heads, head_dim)
        attn_output = attn_output.reshape(batch, seq_len, self.n_q_heads * self.head_dim)
        output = self.o_proj(attn_output)

        return output, new_cache
