"""
RegentModel — the full hybrid Mamba-2 model with dual heads and EPG integration.

Architecture:
    [EPG Encoder] → prefix tokens
    [Essence Vector] → conditioning injection every N layers
    [Token Embedding + Prefix] → Hybrid Backbone → [Gen Head, Ver Head]

Hybrid backbone alternates Mamba-2 layers with sparse GQA attention layers.
The schedule is defined in config (e.g., attention at layers [7, 15, 23, ...]).

Whitepaper alignment:
    - Mamba-2 recurrence = whitepaper state equation s(t+dt) = alpha*s(t) + beta*g(F,c) + gamma*m
    - EPG prefix = whitepaper memory graph nodes injected as context
    - Essence conditioning = whitepaper affective signal modulation
    - Ver Head = grounding confidence (whitepaper confidence metric at token level)
    - Perpetual velocity = the model runs continuously, state persists across calls
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import yaml

from ..blocks.mamba2 import Mamba2Block, Mamba2Config
from ..blocks.attention import GQABlock
from ..blocks.adaptive_gate import AdaptiveGate
from ..heads.gen_head import GenHead
from ..heads.ver_head import VerHead
from ..encoder.epg_encoder import EPGEncoder


@dataclass
class RegentConfig:
    """Full model configuration, loaded from YAML."""

    # Core dimensions
    d_model: int = 1024
    n_layer: int = 48
    vocab_size: int = 32768

    # Mamba-2 SSM
    ssm_expand: int = 2
    ssm_d_state: int = 64
    ssm_d_conv: int = 4
    ssm_n_heads: int = 16
    ssm_chunk_size: int = 64

    # Attention (GQA)
    attn_layers: tuple[int, ...] = ()  # layer indices where GQA is used
    attn_n_q_heads: int = 16
    attn_n_kv_heads: int = 4
    attn_head_dim: int = 64
    attn_window_size: int = 1024

    # Gen head
    tie_embeddings: bool = True

    # Ver head
    ver_enabled: bool = True
    ver_hidden_dim: int = 128

    epg_max_nodes: int = 32
    epg_scalar_features: int = 5
    epg_n_categories: int = 15
    epg_category_embed_dim: int = 8
    epg_n_encoder_layers: int = 2
    epg_encoder_heads: int = 4

    # Essence conditioning
    essence_input_dim: int = 7
    essence_inject_every_n: int = 8

    # Adaptive gate (Phase 5 — disabled by default)
    adaptive_gate: bool = False
    adaptive_gate_hidden: int = 64
    adaptive_gate_threshold: float = 0.5  # hard-gate threshold during inference

    # Norms
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    @classmethod
    def from_yaml(cls, path: str) -> "RegentConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        m = raw["model"]
        ssm = m.get("ssm", {})
        attn = m.get("attention", {})
        gen = m.get("gen_head", {})
        ver = m.get("ver_head", {})
        epg = m.get("epg_encoder", {})
        ess = m.get("essence", {})

        return cls(
            d_model=m["d_model"],
            n_layer=m["n_layer"],
            vocab_size=m["vocab_size"],
            ssm_expand=ssm.get("expand", 2),
            ssm_d_state=ssm.get("d_state", 64),
            ssm_d_conv=ssm.get("d_conv", 4),
            ssm_n_heads=ssm.get("n_heads", 16),
            ssm_chunk_size=ssm.get("chunk_size", 64),
            attn_layers=tuple(attn.get("layers", [])),
            attn_n_q_heads=attn.get("n_q_heads", 16),
            attn_n_kv_heads=attn.get("n_kv_heads", 4),
            attn_head_dim=attn.get("head_dim", 64),
            attn_window_size=attn.get("window_size", 1024),
            tie_embeddings=gen.get("tie_embeddings", True),
            ver_enabled=ver.get("enabled", True),
            ver_hidden_dim=ver.get("hidden_dim", 128),
            epg_max_nodes=epg.get("max_nodes", 32),
            epg_scalar_features=epg.get("scalar_features", 5),
            epg_n_categories=epg.get("n_categories", 15),
            epg_category_embed_dim=epg.get("category_embed_dim", 8),
            epg_n_encoder_layers=epg.get("n_encoder_layers", 2),
            epg_encoder_heads=epg.get("encoder_heads", 4),
            essence_input_dim=ess.get("input_dim", 7),
            essence_inject_every_n=ess.get("inject_every_n", 8),
            adaptive_gate=m.get("adaptive_gate", False),
            adaptive_gate_hidden=m.get("adaptive_gate_hidden", 64),
            adaptive_gate_threshold=m.get("adaptive_gate_threshold", 0.5),
            norm_eps=m.get("norm_eps", 1e-5),
            initializer_range=m.get("initializer_range", 0.02),
        )


class EssenceConditioner(nn.Module):
    """
    Projects the Essence state vector into d_model space for injection.

    Input: 7-dim vector [essence_index, essence_influence, truth_vs_lie,
           civility_vs_unruliness, good_vs_evil, curiosity, self_preservation]
    Output: d_model vector added to hidden states at specified layers.

    Similar to how diffusion models inject timestep embeddings — a conditioning
    signal that shifts behavior without consuming sequence tokens.
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=False),
        )

    def __call__(self, essence: mx.array) -> mx.array:
        """
        Args:
            essence: (batch, input_dim) — essence state vector

        Returns:
            conditioning: (batch, 1, d_model) — additive conditioning
        """
        return self.proj(essence)[:, None, :]


class RegentBlock(nn.Module):
    """
    Single block in the Regent backbone — either Mamba-2, GQA attention,
    or adaptively gated (both paths, learned routing).

    When adaptive_gate is enabled on attention layers, both Mamba and attention
    paths exist. A learned gate decides per-token how much to route through
    attention (precise recall) vs Mamba (local flow).
    """

    def __init__(
        self,
        layer_idx: int,
        cfg: RegentConfig,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_attention = layer_idx in cfg.attn_layers
        self.adaptive = cfg.adaptive_gate and self.is_attention
        self.gate_threshold = cfg.adaptive_gate_threshold

        self.pre_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)

        mamba_cfg = Mamba2Config(
            d_model=cfg.d_model,
            d_state=cfg.ssm_d_state,
            d_conv=cfg.ssm_d_conv,
            expand=cfg.ssm_expand,
            n_heads=cfg.ssm_n_heads,
            norm_eps=cfg.norm_eps,
            chunk_size=cfg.ssm_chunk_size,
        )

        if self.adaptive:
            # Both paths available — gate decides routing
            self.mamba = Mamba2Block(mamba_cfg)
            self.attn = GQABlock(
                d_model=cfg.d_model,
                n_q_heads=cfg.attn_n_q_heads,
                n_kv_heads=cfg.attn_n_kv_heads,
                head_dim=cfg.attn_head_dim,
                window_size=cfg.attn_window_size,
                norm_eps=cfg.norm_eps,
            )
            self.gate = AdaptiveGate(cfg.d_model, cfg.adaptive_gate_hidden)
            self.ff_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
            self.ff = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model * 4, bias=False),
                nn.SiLU(),
                nn.Linear(cfg.d_model * 4, cfg.d_model, bias=False),
            )
        elif self.is_attention:
            self.block = GQABlock(
                d_model=cfg.d_model,
                n_q_heads=cfg.attn_n_q_heads,
                n_kv_heads=cfg.attn_n_kv_heads,
                head_dim=cfg.attn_head_dim,
                window_size=cfg.attn_window_size,
                norm_eps=cfg.norm_eps,
            )
            self.ff_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
            self.ff = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model * 4, bias=False),
                nn.SiLU(),
                nn.Linear(cfg.d_model * 4, cfg.d_model, bias=False),
            )
        else:
            self.block = Mamba2Block(mamba_cfg)

    def __call__(
        self,
        x: mx.array,
        essence_cond: mx.array | None = None,
        cache: dict | None = None,
        use_chunked: bool = True,
    ) -> tuple[mx.array, dict | None]:
        h = self.pre_norm(x)

        if self.adaptive:
            # Learned routing between Mamba and attention
            gate = self.gate(h)  # (batch, seq_len, 1)

            mamba_out, mamba_cache = self.mamba(h, cache=cache.get("mamba") if cache else None, use_chunked=use_chunked)
            attn_out, attn_cache = self.attn(h, cache=cache.get("attn") if cache else None)

            # Soft mixture during training, hard gate during inference
            if use_chunked:
                block_out = gate * attn_out + (1.0 - gate) * mamba_out
            else:
                hard_gate = (gate > self.gate_threshold).astype(mx.float32)
                block_out = hard_gate * attn_out + (1.0 - hard_gate) * mamba_out

            x = x + block_out
            x = x + self.ff(self.ff_norm(x))

            new_cache = {"mamba": mamba_cache, "attn": attn_cache} if cache is not None else None

        elif self.is_attention:
            block_out, new_cache = self.block(h, cache=cache)
            x = x + block_out
            x = x + self.ff(self.ff_norm(x))

        else:
            block_out, new_cache = self.block(h, cache=cache, use_chunked=use_chunked)
            x = x + block_out

        if essence_cond is not None:
            x = x + essence_cond

        return x, new_cache


class RegentModel(nn.Module):
    """
    The full Regent model.

    Components:
        - Token embedding (shared with Gen Head if tie_embeddings=True)
        - EPG encoder (knowledge graph → prefix tokens)
        - Essence conditioner (affective state → layer injection)
        - Hybrid backbone (Mamba-2 + sparse GQA)
        - Gen Head (next-token prediction)
        - Ver Head (grounding score prediction)
    """

    def __init__(self, cfg: RegentConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.epg_encoder = EPGEncoder(
            d_model=cfg.d_model,
            scalar_features=cfg.epg_scalar_features,
            n_categories=cfg.epg_n_categories,
            category_embed_dim=cfg.epg_category_embed_dim,
            n_encoder_layers=cfg.epg_n_encoder_layers,
            encoder_heads=cfg.epg_encoder_heads,
            norm_eps=cfg.norm_eps,
        )

        self.essence_cond = EssenceConditioner(cfg.essence_input_dim, cfg.d_model)

        self.layers = [RegentBlock(i, cfg) for i in range(cfg.n_layer)]

        self.gen_head = GenHead(cfg.d_model, cfg.vocab_size, cfg.norm_eps)

        if cfg.ver_enabled:
            self.ver_head = VerHead(cfg.d_model, cfg.ver_hidden_dim, cfg.norm_eps)
        else:
            self.ver_head = None

        if cfg.tie_embeddings:
            self.gen_head.tie_weights(self.embed.weight)

    def _build_prefix(
        self,
        input_ids: mx.array,
        epg_node_tokens: mx.array | None = None,
        epg_scalars: mx.array | None = None,
        epg_categories: mx.array | None = None,
    ) -> mx.array:
        """
        Build the full input embedding sequence:
        [EPG prefix tokens] [token embeddings]
        """
        token_embeds = self.embed(input_ids)  # (batch, seq_len, d_model)

        if epg_node_tokens is not None and epg_scalars is not None and epg_categories is not None:
            epg_token_embeds = self.embed(epg_node_tokens)  # (batch, n_nodes, max_node_tokens, d_model)

            prefix = self.epg_encoder(
                epg_token_embeds, epg_scalars, epg_categories
            )  # (batch, n_nodes, d_model)

            token_embeds = mx.concatenate([prefix, token_embeds], axis=1)

        return token_embeds

    def backbone(
        self,
        x: mx.array,
        essence: mx.array | None = None,
        cache: list[dict] | None = None,
        use_chunked: bool = True,
    ) -> tuple[mx.array, list[dict] | None]:
        """
        Run the hybrid backbone (all layers).

        Args:
            x: (batch, seq_len, d_model) — embedded input
            essence: (batch, essence_dim) — essence state vector
            cache: list of per-layer cache dicts for inference
            use_chunked: use SSD parallel form (True for training)

        Returns:
            h: (batch, seq_len, d_model) — final hidden states
            new_caches: updated per-layer caches
        """
        cfg = self.cfg

        ess_cond = self.essence_cond(essence) if essence is not None else None

        new_caches = [] if cache is not None else None

        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None

            # Only inject essence every N layers
            inject = ess_cond if (i % cfg.essence_inject_every_n == 0) else None

            x, new_layer_cache = layer(
                x, essence_cond=inject, cache=layer_cache, use_chunked=use_chunked
            )

            if new_caches is not None:
                new_caches.append(new_layer_cache or {})

        return x, new_caches

    def __call__(
        self,
        input_ids: mx.array,
        essence: mx.array | None = None,
        epg_node_tokens: mx.array | None = None,
        epg_scalars: mx.array | None = None,
        epg_categories: mx.array | None = None,
        cache: list[dict] | None = None,
        use_chunked: bool = True,
    ) -> dict:
        """
        Full forward pass.

        Args:
            input_ids: (batch, seq_len) — token IDs
            essence: (batch, 7) — essence state vector
            epg_node_tokens: (batch, n_nodes, max_node_tokens) — EPG text token IDs
            epg_scalars: (batch, n_nodes, n_scalar_features) — EPG scalar features
            epg_categories: (batch, n_nodes) — EPG category IDs
            cache: per-layer cache for inference
            use_chunked: use SSD parallel form for Mamba layers

        Returns:
            dict with:
                logits: (batch, seq_len, vocab_size) — next-token logits
                grounding: (batch, seq_len) — verification scores (if ver_head enabled)
                hidden: (batch, seq_len, d_model) — final hidden states
                cache: updated per-layer caches
        """
        x = self._build_prefix(input_ids, epg_node_tokens, epg_scalars, epg_categories)

        h, new_caches = self.backbone(x, essence=essence, cache=cache, use_chunked=use_chunked)

        # If EPG prefix was prepended, slice it off for head computation
        # (the prefix influenced the backbone but we don't need logits for it)
        if epg_node_tokens is not None:
            h_for_heads = h[:, epg_node_tokens.shape[1]:]
        else:
            h_for_heads = h

        logits = self.gen_head(h_for_heads)

        result = {
            "logits": logits,
            "hidden": h_for_heads,
            "cache": new_caches,
        }

        if self.ver_head is not None:
            result["grounding"] = self.ver_head(h_for_heads)

        return result

    def init_cache(self) -> list[dict]:
        """Initialize empty caches for each layer (inference mode)."""
        return [{} for _ in range(self.cfg.n_layer)]

    def count_parameters(self) -> dict:
        """Count total parameters."""
        def _count(params):
            if isinstance(params, mx.array):
                return params.size
            if isinstance(params, dict):
                return sum(_count(v) for v in params.values())
            if isinstance(params, list):
                return sum(_count(v) for v in params)
            return 0

        total = _count(self.parameters())
        return {"total": total, "total_millions": round(total / 1e6, 1)}
