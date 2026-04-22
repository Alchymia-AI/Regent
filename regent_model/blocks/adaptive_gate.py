"""
Adaptive attention gate.

Learns per-token whether to route through attention (precise recall) or
let Mamba handle it (compressed local flow). Produces a scalar gate in [0, 1]:
    gate → 1: attention needed (long-range dependency)
    gate → 0: Mamba sufficient (local continuation)

During training, uses a soft mixture of both paths so gradients flow
through both. During inference, hard-gates at a threshold to save compute.

This is a Phase 5 addition. Disabled by default. Enable with
adaptive_gate: true in the YAML config.
"""

import mlx.core as mx
import mlx.nn as nn


class AdaptiveGate(nn.Module):
    """
    Learns when attention is needed vs when Mamba is sufficient.

    Architecture:
        hidden → LayerNorm → Linear → SiLU → Linear → sigmoid → scalar gate

    Small: 2 * d_model * gate_hidden parameters per gated layer.
    """

    def __init__(self, d_model: int, hidden_dim: int = 64):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            gate: (batch, seq_len, 1) — values in [0, 1]
                1.0 = route through attention
                0.0 = route through Mamba only
        """
        return mx.sigmoid(self.proj(self.norm(x)))
