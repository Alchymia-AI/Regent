"""
Verification head — grounding score predictor for hallucination mitigation.

Reads the same hidden states as the Gen head but outputs a scalar grounding
score per token position. Trained to predict whether generated content is
factually grounded (1.0) or fabricated (0.0).

This implements the Regent's Option 2 hallucination architecture:
the Ver head monitors the backbone's hidden state for entropy/uncertainty.
When grounding drops below threshold, the decoding loop can:
  - Lower temperature (CAUTION zone)
  - Hedge with uncertainty language
  - Halt and trigger EPG retrieval (HALT zone)

Architecturally identical to RLHF reward model heads — shared backbone,
separate scalar projection. Well-established pattern, novel application.
"""

import mlx.core as mx
import mlx.nn as nn


class VerHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 128, norm_eps: float = 1e-5):
        super().__init__()
        self.norm = nn.RMSNorm(d_model, eps=norm_eps)
        self.fc1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        """
        Args:
            h: (batch, seq_len, d_model) — hidden states from backbone

        Returns:
            grounding_scores: (batch, seq_len) — values in [0, 1]
                1.0 = grounded/factual
                0.0 = fabricated/hallucinated
        """
        x = self.norm(h)
        x = nn.silu(self.fc1(x))
        x = mx.sigmoid(self.fc2(x))
        return x.squeeze(-1)  # (batch, seq_len)
