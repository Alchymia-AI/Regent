"""
Generation head — standard autoregressive language model head.

Projects hidden states to vocabulary logits for next-token prediction.
Optionally ties weights with the input embedding matrix.
"""

import mlx.core as mx
import mlx.nn as nn


class GenHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def tie_weights(self, embedding_weight: mx.array):
        """Tie output projection weights with input embeddings."""
        self.proj.weight = embedding_weight

    def __call__(self, h: mx.array) -> mx.array:
        """
        Args:
            h: (batch, seq_len, d_model) — hidden states from backbone

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        return self.proj(self.norm(h))
