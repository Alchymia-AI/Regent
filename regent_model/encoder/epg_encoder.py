"""
EPG Encoder — converts Entity Preference Graph nodes into dense prefix embeddings.

Instead of serializing EPG nodes as markdown text (burning 500-1000 context tokens),
this encoder produces dense embedding vectors prepended as virtual tokens.

Each EPG node has:
    - key + value (text, tokenized)
    - confidence (float, 0-1)
    - activation / rho (float, 0-1, decays per whitepaper)
    - valence (float, -1 to +1)
    - emotional_weight (float)
    - category (enum, 15 types from whitepaper)

The encoder processes text features through a small transformer and concatenates
with projected scalar features, producing one d_model embedding per EPG node.
These embeddings are prepended to the token sequence as "virtual tokens" that
the backbone attends to naturally.

Reference: Prefix-Tuning (Li & Liang, 2021) — virtual token injection is
well-established. Applying it to structured KG nodes is the novel aspect.
"""

import mlx.core as mx
import mlx.nn as nn


class EPGNodeEncoder(nn.Module):
    """Encodes a single EPG node's text content into a fixed-size vector."""

    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 4, norm_eps: float = 1e-5):
        super().__init__()
        self.layers = [
            EPGEncoderLayer(d_model, n_heads, norm_eps) for _ in range(n_layers)
        ]
        self.pool_norm = nn.RMSNorm(d_model, eps=norm_eps)

    def __call__(self, token_embeds: mx.array) -> mx.array:
        """
        Args:
            token_embeds: (batch, n_nodes, max_node_tokens, d_model)
                Embedded tokens for each node's key+value text

        Returns:
            node_embeds: (batch, n_nodes, d_model)
                One embedding per node (mean-pooled over tokens)
        """
        batch, n_nodes, max_tokens, d_model = token_embeds.shape

        # Flatten batch and nodes for processing
        x = token_embeds.reshape(batch * n_nodes, max_tokens, d_model)

        for layer in self.layers:
            x = layer(x)

        # Mean pool over token dimension
        x = x.mean(axis=1)  # (batch * n_nodes, d_model)
        x = self.pool_norm(x)

        return x.reshape(batch, n_nodes, d_model)


class EPGEncoderLayer(nn.Module):
    """Single transformer encoder layer for EPG text processing."""

    def __init__(self, d_model: int, n_heads: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=norm_eps)
        self.norm2 = nn.RMSNorm(d_model, eps=norm_eps)
        self.attn = nn.MultiHeadAttention(d_model, n_heads, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model, bias=False),
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class EPGEncoder(nn.Module):
    """
    Full EPG encoder: text features + scalar features -> prefix embeddings.

    Whitepaper alignment:
        - confidence -> maps to node reliability for response grounding
        - activation (rho) -> decays by alpha each cycle, drives recency
        - valence -> feel-good/feel-bad signal from belief clusters
        - emotional_weight -> cluster membership weight (w_j in sigma_k formula)
        - category -> one of 15 types (identity, belief, capability, etc.)
    """

    def __init__(
        self,
        d_model: int,
        scalar_features: int = 5,
        n_categories: int = 15,
        category_embed_dim: int = 8,
        n_encoder_layers: int = 2,
        encoder_heads: int = 4,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        # Text encoder (processes tokenized key+value)
        self.text_encoder = EPGNodeEncoder(
            d_model, n_layers=n_encoder_layers, n_heads=encoder_heads, norm_eps=norm_eps
        )

        # Scalar feature projection
        # scalar_features: [confidence, activation, valence, emotional_weight]
        # + category embedding
        self.category_embed = nn.Embedding(n_categories, category_embed_dim)
        scalar_total = scalar_features + category_embed_dim
        self.scalar_proj = nn.Linear(scalar_total, d_model, bias=False)

        # Fusion: combine text embedding + scalar projection
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

        self.output_norm = nn.RMSNorm(d_model, eps=norm_eps)

    def __call__(
        self,
        node_token_embeds: mx.array,
        scalar_features: mx.array,
        category_ids: mx.array,
    ) -> mx.array:
        """
        Args:
            node_token_embeds: (batch, n_nodes, max_node_tokens, d_model)
                Token embeddings for each node's key+value text.
                Uses the main model's embedding layer.

            scalar_features: (batch, n_nodes, n_scalar_features)
                [confidence, activation, valence, emotional_weight, ...]

            category_ids: (batch, n_nodes)
                Integer category IDs (0-14 for the 15 EPG categories)

        Returns:
            prefix_embeds: (batch, n_nodes, d_model)
                Dense embeddings to prepend to the token sequence
        """
        # Encode text
        text_embeds = self.text_encoder(node_token_embeds)  # (batch, n_nodes, d_model)

        # Encode scalars + category
        cat_embeds = self.category_embed(category_ids)  # (batch, n_nodes, category_embed_dim)
        scalars = mx.concatenate([scalar_features, cat_embeds], axis=-1)
        scalar_embeds = self.scalar_proj(scalars)  # (batch, n_nodes, d_model)

        # Fuse text and scalar features
        combined = mx.concatenate([text_embeds, scalar_embeds], axis=-1)  # (batch, n_nodes, 2*d_model)
        fused = self.fusion(combined)  # (batch, n_nodes, d_model)

        return self.output_norm(fused)
