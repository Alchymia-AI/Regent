"""
PyTorch implementation of Regent for HuggingFace Transformers.

Runs on CPU, CUDA, and Apple MPS — move the model with:
    model.to("cuda")   # NVIDIA GPU
    model.to("mps")    # Apple Silicon

Weight keys mirror the MLX training implementation exactly so safetensors
weights load without renaming.

Usage
-----
    from transformers import AutoConfig, AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
                "path/to/export",
                trust_remote_code=True,
                torch_dtype=torch.float16,   # float32 / float16 / bfloat16
                device_map="auto",           # or "cuda" / "mps" / "cpu"
            )
    outputs = model.generate(input_ids, max_new_tokens=256)

Optional: compile for extra speed on CUDA (requires PyTorch >= 2.0):
    model = torch.compile(model)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_regent import RegentConfig

# Optional fast path: mamba-ssm Triton SSD kernel (CUDA only)
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _mamba_ssd_fn
    _MAMBA_SSM_AVAILABLE = True
except ImportError:
    _MAMBA_SSM_AVAILABLE = False


# ---------------------------------------------------------------------------
# Naming-compatible building blocks
# (attribute names identical to MLX so safetensors keys match)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    """RMSNorm — `weight` attribute matches mlx.nn.RMSNorm."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class _NoParam(nn.Module):
    """Parameter-less SiLU placeholder, keeping Sequential index alignment."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class _Sequential(nn.Module):
    """
    Sequential with submodules in `self.layers` (nn.ModuleList).
    Mirrors mlx.nn.Sequential which stores `self.layers = [...]` so
    parameter key prefixes are identical, e.g.:
        essence_cond.proj.layers.0.weight
        essence_cond.proj.layers.2.weight
    """

    def __init__(self, *modules):
        super().__init__()
        self.layers = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.layers:
            x = m(x)
        return x


class _MultiHeadAttn(nn.Module):
    """
    MHA with projection names matching mlx.nn.MultiHeadAttention:
    query_proj, key_proj, value_proj, out_proj.
    """

    def __init__(self, d_model: int, n_heads: int, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.query_proj = nn.Linear(d_model, d_model, bias=bias)
        self.key_proj   = nn.Linear(d_model, d_model, bias=bias)
        self.value_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj   = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, Lq, D = q.shape
        Lk = k.shape[1]
        q = self.query_proj(q).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key_proj(k).view(  B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value_proj(v).view( B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        return self.out_proj((attn @ v).transpose(1, 2).reshape(B, Lq, D))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class EPGEncoderLayer(nn.Module):

    def __init__(self, d_model: int, n_heads: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm1 = _RMSNorm(d_model, eps=norm_eps)
        self.norm2 = _RMSNorm(d_model, eps=norm_eps)
        self.attn  = _MultiHeadAttn(d_model, n_heads, bias=False)
        self.ff    = _Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            _NoParam(),
            nn.Linear(d_model * 4, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        x = x + self.attn(n, n, n)
        return x + self.ff(self.norm2(x))


class EPGNodeEncoder(nn.Module):

    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 4, norm_eps: float = 1e-5):
        super().__init__()
        self.layers    = nn.ModuleList([EPGEncoderLayer(d_model, n_heads, norm_eps) for _ in range(n_layers)])
        self.pool_norm = _RMSNorm(d_model, eps=norm_eps)

    def forward(self, token_embeds: torch.Tensor) -> torch.Tensor:
        B, N, T, D = token_embeds.shape
        x = token_embeds.view(B * N, T, D)
        for layer in self.layers:
            x = layer(x)
        return self.pool_norm(x.mean(dim=1)).view(B, N, D)


class EPGEncoder(nn.Module):

    def __init__(self, d_model: int, scalar_features: int = 5, n_categories: int = 15,
                 category_embed_dim: int = 8, n_encoder_layers: int = 2,
                 encoder_heads: int = 4, norm_eps: float = 1e-5):
        super().__init__()
        self.text_encoder   = EPGNodeEncoder(d_model, n_encoder_layers, encoder_heads, norm_eps)
        self.category_embed = nn.Embedding(n_categories, category_embed_dim)
        self.scalar_proj    = nn.Linear(scalar_features + category_embed_dim, d_model, bias=False)
        self.fusion         = _Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            _NoParam(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.output_norm = _RMSNorm(d_model, eps=norm_eps)

    def forward(self, node_token_embeds: torch.Tensor,
                scalar_features: torch.Tensor,
                category_ids: torch.LongTensor) -> torch.Tensor:
        text    = self.text_encoder(node_token_embeds)
        cat     = self.category_embed(category_ids)
        scalars = self.scalar_proj(torch.cat([scalar_features, cat], dim=-1))
        fused   = self.fusion(torch.cat([text, scalars], dim=-1))
        return self.output_norm(fused)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class EssenceConditioner(nn.Module):

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.proj = _Sequential(
            nn.Linear(input_dim, d_model, bias=False),
            _NoParam(),
            nn.Linear(d_model, d_model, bias=False),
        )

    def forward(self, essence: torch.Tensor) -> torch.Tensor:
        return self.proj(essence).unsqueeze(1)   # (B, 1, d_model)


# ---------------------------------------------------------------------------
# Mamba-2 block  — GPU-accelerated conv + parallel scan
# ---------------------------------------------------------------------------

class Mamba2Block(nn.Module):
    """
    Mamba-2 SSM block.

    Three computation paths:
      • _scan_step   — single token update for cached generation (O(1) state)
      • _scan_triton — mamba-ssm Triton SSD kernel (CUDA, fastest when available)
      • _scan_torch  — vectorized parallel prefix scan (CUDA/MPS/CPU fallback)

    _scan_prefill dispatches to Triton when mamba-ssm is installed and the
    input is on CUDA; otherwise falls back to the pure-PyTorch implementation.
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, n_heads: int = 16, norm_eps: float = 1e-5,
                 chunk_size: int = 64):
        super().__init__()
        self.d_model    = d_model
        self.d_state    = d_state
        self.d_conv     = d_conv
        self.n_heads    = n_heads
        self.chunk_size = chunk_size

        self.d_inner  = d_model * expand
        self.head_dim = self.d_inner // n_heads
        assert self.d_inner % n_heads == 0

        in_proj_dim = self.d_inner * 2 + n_heads * d_state * 2 + n_heads
        self.in_proj = nn.Linear(d_model, in_proj_dim, bias=False)

        # Plain Parameters — same names as MLX raw array attributes
        self.conv1d_weight = nn.Parameter(torch.zeros(self.d_inner, d_conv))
        self.conv1d_bias   = nn.Parameter(torch.zeros(self.d_inner))
        self.A_log         = nn.Parameter(torch.full((n_heads,), math.log(0.5)))
        self.D             = nn.Parameter(torch.ones(n_heads))
        self.dt_bias       = nn.Parameter(torch.zeros(n_heads))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = _RMSNorm(self.d_inner, eps=norm_eps)

    # ------------------------------------------------------------------ conv

    def _causal_conv1d(
        self, x: torch.Tensor, cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GPU-accelerated depthwise causal conv1d via F.conv1d.

        x:     (B, L, d_inner)
        cache: (B, d_conv-1, d_inner) — context from previous step, or None
        Returns y (B, L, d_inner), new_cache (B, d_conv-1, d_inner)
        """
        B, L, D = x.shape
        k = self.d_conv

        # (B, d_inner, L)
        x_t = x.transpose(1, 2)

        if cache is not None:
            # cache: (B, d_conv-1, d_inner) → (B, d_inner, d_conv-1)
            x_t = torch.cat([cache.transpose(1, 2), x_t], dim=2)
        else:
            x_t = F.pad(x_t, (k - 1, 0))

        # Depthwise: weight (d_inner, 1, d_conv), groups=d_inner
        w   = self.conv1d_weight.unsqueeze(1)           # (d_inner, 1, d_conv)
        y   = F.conv1d(x_t, w, bias=self.conv1d_bias, groups=D)  # (B, d_inner, L)

        new_cache = x_t[:, :, -(k - 1):].transpose(1, 2).detach()  # (B, d_conv-1, d_inner)
        return y.transpose(1, 2), new_cache              # (B, L, d_inner)

    # ------------------------------------------------------------------ scan: prefill (dispatcher)

    def _scan_prefill(
        self,
        x_ssm:     torch.Tensor,
        A:         torch.Tensor,
        B:         torch.Tensor,
        C:         torch.Tensor,
        dt:        torch.Tensor,
        ssm_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Route to Triton SSD kernel on CUDA (if mamba-ssm installed), else PyTorch."""
        if _MAMBA_SSM_AVAILABLE and x_ssm.is_cuda:
            return self._scan_triton(x_ssm, A, B, C, dt, ssm_state)
        return self._scan_torch(x_ssm, A, B, C, dt, ssm_state)

    def _scan_triton(
        self,
        x_ssm:     torch.Tensor,       # (B, L, nh, hd)
        A:         torch.Tensor,       # (nh,)
        B:         torch.Tensor,       # (B, L, nh, ds)
        C:         torch.Tensor,       # (B, L, nh, ds)
        dt:        torch.Tensor,       # (B, L, nh)
        ssm_state: Optional[torch.Tensor] = None,  # (B, nh, hd, ds)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast path: mamba-ssm Triton SSD kernel (CUDA only).

        mamba_chunk_scan_combined expects:
          x:  (B, L, nh, hd)   dt: (B, L, nh)   A: (nh,)
          B:  (B, L, ngroups, ds)  C: (B, L, ngroups, ds)
          D:  (nh,)  initial_states: (B, nh, hd, ds)
        All tensors must be contiguous and on CUDA.
        """
        y, final_state = _mamba_ssd_fn(
            x_ssm.contiguous(),
            dt.contiguous(),
            A.contiguous(),
            B.contiguous(),
            C.contiguous(),
            chunk_size=self.chunk_size,
            D=self.D.contiguous(),
            initial_states=(ssm_state.contiguous() if ssm_state is not None else None),
            dt_softplus=False,          # softplus already applied upstream
            return_final_states=True,
        )
        return y, final_state

    def _scan_torch(
        self,
        x_ssm:     torch.Tensor,       # (B, L, nh, hd)
        A:         torch.Tensor,       # (nh,)
        B:         torch.Tensor,       # (B, L, nh, ds)
        C:         torch.Tensor,       # (B, L, nh, ds)
        dt:        torch.Tensor,       # (B, L, nh)
        ssm_state: Optional[torch.Tensor] = None,  # (B, nh, hd, ds)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized parallel prefix scan — no Python loops.

        Solves h_t = A_t·h_{t-1} + (dt_t·B_t)⊗x_t using the identity:

            h_t = exp(Φ_t) · (h_{-1} + Σ_{s≤t} exp(-Φ_s)·b_s)

        where Φ_t = cumsum(A·dt)[t] and b_s = (dt_s·B_s)⊗x_s.

        All ops are element-wise or cumsum — runs entirely on CUDA/MPS.
        """
        B_sz, L, nh, hd = x_ssm.shape
        ds = B.shape[-1]

        # Log decay per step: (B, L, nh)
        log_decay = A[None, None, :] * dt   # A is negative → log_decay < 0

        # Cumulative log decay Φ[t] = Σ_{k≤t} log_decay[k] ≤ 0
        Phi = torch.cumsum(log_decay, dim=1)   # (B, L, nh)

        # b_t = (dt_t · B_t) ⊗ x_t  →  (B, L, nh, hd, ds)
        dB = dt.unsqueeze(-1) * B              # (B, L, nh, ds)
        b  = dB.unsqueeze(3) * x_ssm.unsqueeze(4)  # (B, L, nh, hd, ds)

        # Clamp to prevent exp overflow in float16 / bfloat16
        max_exp = 80.0 if x_ssm.dtype == torch.float32 else 10.0

        # exp(-Φ): (B, L, nh, 1, 1) — weights to scale b down
        exp_neg_Phi = torch.exp((-Phi).clamp(max=max_exp)).unsqueeze(3).unsqueeze(4)
        b_scaled    = exp_neg_Phi * b                    # (B, L, nh, hd, ds)
        b_cumsum    = torch.cumsum(b_scaled, dim=1)      # (B, L, nh, hd, ds)

        # Add initial state (if cache present): h_{-1} contributes exp(Φ_t)·h_{-1}
        if ssm_state is not None:
            b_cumsum = b_cumsum + ssm_state.unsqueeze(1)  # (B, L, nh, hd, ds)

        # exp(Φ): Φ ≤ 0 so exp(Φ) ∈ (0, 1] — no overflow
        exp_Phi = torch.exp(Phi).unsqueeze(3).unsqueeze(4)
        h = exp_Phi * b_cumsum                           # (B, L, nh, hd, ds)

        # Output: y_t = Σ_s C_t[s]·h_t[:,s] + D[h]·x_t
        y = (C.unsqueeze(3) * h).sum(-1)                 # (B, L, nh, hd)
        y = y + self.D[None, None, :, None] * x_ssm      # skip connection

        return y, h[:, -1]   # (B, L, nh, hd), final_state (B, nh, hd, ds)

    # ------------------------------------------------------------------ scan: single step

    def _scan_step(
        self,
        x_ssm:     torch.Tensor,     # (B, 1, nh, hd)
        A:         torch.Tensor,     # (nh,)
        B:         torch.Tensor,     # (B, 1, nh, ds)
        C:         torch.Tensor,     # (B, 1, nh, ds)
        dt:        torch.Tensor,     # (B, 1, nh)
        ssm_state: torch.Tensor,     # (B, nh, hd, ds)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        O(1) single-step SSM update for cached autoregressive generation.
        No scan needed — just one recurrence step.
        """
        dt_t = dt[:, 0]                                   # (B, nh)
        B_t  = B[:, 0]                                    # (B, nh, ds)
        C_t  = C[:, 0]                                    # (B, nh, ds)
        x_t  = x_ssm[:, 0]                               # (B, nh, hd)

        dA = torch.exp(A[None, :] * dt_t)                # (B, nh)
        dB = dt_t.unsqueeze(-1) * B_t                    # (B, nh, ds)

        # State update: h = dA·h + (dB⊗x)
        h  = (dA.unsqueeze(-1).unsqueeze(-1) * ssm_state
              + dB.unsqueeze(2) * x_t.unsqueeze(3))      # (B, nh, hd, ds)

        # Output: y = Σ_s C[s]·h[:,s] + D·x
        y_t = (C_t.unsqueeze(2) * h).sum(-1)             # (B, nh, hd)
        y_t = y_t + self.D[:, None] * x_t

        return y_t.unsqueeze(1), h   # (B, 1, nh, hd), (B, nh, hd, ds)

    # ------------------------------------------------------------------ forward

    def forward(
        self, x: torch.Tensor, cache: Optional[dict] = None, **_
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        B_sz, L, _ = x.shape
        nh, ds = self.n_heads, self.d_state

        # Input projection
        proj  = self.in_proj(x)
        idx   = 0
        z     = proj[:, :, idx : idx + self.d_inner]; idx += self.d_inner
        x_ssm = proj[:, :, idx : idx + self.d_inner]; idx += self.d_inner
        Bm    = proj[:, :, idx : idx + nh * ds];      idx += nh * ds
        Cm    = proj[:, :, idx : idx + nh * ds];      idx += nh * ds
        dt_r  = proj[:, :, idx : idx + nh]

        Bm = Bm.view(B_sz, L, nh, ds)
        Cm = Cm.view(B_sz, L, nh, ds)
        dt = F.softplus(dt_r) + self.dt_bias[None, None, :]  # (B, L, nh)

        # Depthwise causal conv1d — GPU-native via F.conv1d
        conv_cache        = cache.get("conv") if cache else None
        x_ssm, new_conv   = self._causal_conv1d(x_ssm, conv_cache)
        x_ssm             = F.silu(x_ssm)

        x_ssm = x_ssm.view(B_sz, L, nh, self.head_dim)
        A     = -torch.exp(self.A_log)          # (nh,) — negative decay

        ssm_state = cache.get("ssm") if cache else None

        # Dispatch: single-step path for cached generation, scan for prefill
        if L == 1 and ssm_state is not None:
            y, new_ssm = self._scan_step(x_ssm, A, Bm, Cm, dt, ssm_state)
        else:
            y, new_ssm = self._scan_prefill(x_ssm, A, Bm, Cm, dt, ssm_state)

        y      = y.reshape(B_sz, L, self.d_inner)
        y      = self.norm(y) * F.silu(z)
        output = self.out_proj(y)

        new_cache = {"conv": new_conv, "ssm": new_ssm} if cache is not None else None
        return output, new_cache


# ---------------------------------------------------------------------------
# GQA attention block
# ---------------------------------------------------------------------------

class GQABlock(nn.Module):
    """GQA with RoPE and sliding-window KV cache."""

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int,
                 head_dim: int, window_size: int = 1024, norm_eps: float = 1e-5):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.n_q_heads   = n_q_heads
        self.n_kv_heads  = n_kv_heads
        self.head_dim    = head_dim
        self.window_size = window_size
        self.n_groups    = n_q_heads // n_kv_heads
        self.scale       = head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, n_q_heads  * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_q_heads * head_dim, d_model,  bias=False)
        self.norm   = _RMSNorm(d_model, eps=norm_eps)

    def _rope(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        _, L, _, hd = x.shape
        inv = 1.0 / (10000.0 ** (torch.arange(0, hd, 2, dtype=torch.float32, device=x.device) / hd))
        pos = torch.arange(offset, offset + L, dtype=torch.float32, device=x.device)
        f   = pos[:, None] * inv[None, :]
        c, s = f.cos(), f.sin()
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * c[None, :, None, :] - x2 * s[None, :, None, :],
                             x1 * s[None, :, None, :] + x2 * c[None, :, None, :]], dim=-1).reshape(x.shape)

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None) -> Tuple[torch.Tensor, dict]:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_q_heads,  self.head_dim)
        k = self.k_proj(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.n_kv_heads, self.head_dim)

        off = cache["k"].shape[1] if (cache and "k" in cache) else 0
        q, k = self._rope(q, off), self._rope(k, off)

        if cache and "k" in cache:
            k = torch.cat([cache["k"], k], dim=1)
            v = torch.cat([cache["v"], v], dim=1)
            if k.shape[1] > self.window_size:
                k, v = k[:, -self.window_size:], v[:, -self.window_size:]

        kv_len = k.shape[1]
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=2)
            v = v.repeat_interleave(self.n_groups, dim=2)

        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = torch.tril(torch.ones(L, kv_len, device=x.device), diagonal=kv_len - L)
        attn = attn.masked_fill(mask[None, None] == 0, float("-inf"))
        out  = (torch.softmax(attn, dim=-1) @ v).transpose(1, 2).reshape(B, L, -1)
        return self.o_proj(out), {"k": k.transpose(1, 2), "v": v.transpose(1, 2)}


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

class GenHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm = _RMSNorm(d_model, eps=norm_eps)
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(h))


class VerHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 128, norm_eps: float = 1e-5):
        super().__init__()
        self.norm = _RMSNorm(d_model, eps=norm_eps)
        self.fc1  = nn.Linear(d_model, hidden_dim, bias=False)
        self.fc2  = nn.Linear(hidden_dim, 1,        bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc2(F.silu(self.fc1(self.norm(h))))).squeeze(-1)


# ---------------------------------------------------------------------------
# Backbone block
# ---------------------------------------------------------------------------

class RegentBlock(nn.Module):
    """Mamba-2 or GQA block with pre-norm, optional FFN, essence injection."""

    def __init__(self, layer_idx: int, cfg: RegentConfig):
        super().__init__()
        self.is_attention = layer_idx in list(cfg.attn_layers)
        self.pre_norm     = _RMSNorm(cfg.d_model, eps=cfg.norm_eps)

        if self.is_attention:
            self.block    = GQABlock(cfg.d_model, cfg.attn_n_q_heads, cfg.attn_n_kv_heads,
                                     cfg.attn_head_dim, cfg.attn_window_size, cfg.norm_eps)
            self.ff_norm  = _RMSNorm(cfg.d_model, eps=cfg.norm_eps)
            self.ff       = _Sequential(
                nn.Linear(cfg.d_model, cfg.d_model * 4, bias=False),
                _NoParam(),
                nn.Linear(cfg.d_model * 4, cfg.d_model, bias=False),
            )
        else:
            self.block = Mamba2Block(cfg.d_model, cfg.ssm_d_state, cfg.ssm_d_conv,
                                     cfg.ssm_expand, cfg.ssm_n_heads, cfg.norm_eps,
                                     cfg.ssm_chunk_size)

    def forward(self, x: torch.Tensor, essence_cond: Optional[torch.Tensor] = None,
                cache: Optional[dict] = None, **_) -> Tuple[torch.Tensor, Optional[dict]]:
        h             = self.pre_norm(x)
        block_out, nc = self.block(h, cache=cache)
        x             = x + block_out
        if self.is_attention:
            x = x + self.ff(self.ff_norm(x))
        if essence_cond is not None:
            x = x + essence_cond
        return x, nc


# ---------------------------------------------------------------------------
# Top-level causal LM model
# ---------------------------------------------------------------------------

class RegentForCausalLM(PreTrainedModel):
    """
    Regent causal LM — HuggingFace PreTrainedModel.

    Top-level attribute names match the MLX RegentModel exactly:
        embed, epg_encoder, essence_cond, layers, gen_head, ver_head

    Device placement:
        model.to("cuda")  →  all ops on CUDA
        model.to("mps")   →  all ops on Apple Metal
        model.to("cpu")   →  CPU fallback
    """

    config_class                       = RegentConfig
    _keys_to_ignore_on_load_missing    = [r"gen_head\.proj\.weight"]
    _keys_to_ignore_on_load_unexpected = []

    def __init__(self, config: RegentConfig):
        super().__init__(config)
        self.embed        = nn.Embedding(config.vocab_size, config.d_model)
        self.epg_encoder  = EPGEncoder(
            config.d_model, config.epg_scalar_features, config.epg_n_categories,
            config.epg_category_embed_dim, config.epg_n_encoder_layers,
            config.epg_encoder_heads, config.norm_eps,
        )
        self.essence_cond = EssenceConditioner(config.essence_input_dim, config.d_model)
        self.layers       = nn.ModuleList([RegentBlock(i, config) for i in range(config.n_layer)])
        self.gen_head     = GenHead(config.d_model, config.vocab_size, config.norm_eps)
        self.ver_head: Optional[VerHead] = (
            VerHead(config.d_model, config.ver_hidden_dim, config.norm_eps)
            if config.ver_enabled else None
        )
        self.post_init()

    # ---- weight tying -------------------------------------------------------

    def get_input_embeddings(self)        -> nn.Embedding: return self.embed
    def set_input_embeddings(self, v)     -> None:          self.embed = v
    def get_output_embeddings(self)       -> nn.Linear:     return self.gen_head.proj
    def set_output_embeddings(self, v)    -> None:          self.gen_head.proj = v

    def tie_weights(self) -> None:
        if self.config.tie_embeddings:
            self.gen_head.proj.weight = self.embed.weight

    # ---- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids:        torch.LongTensor,
        attention_mask:   Optional[torch.Tensor]         = None,
        past_key_values:  Optional[List[Optional[dict]]] = None,
        use_cache:        bool                           = True,
        return_dict:      bool                           = True,
        # Regent-specific optional inputs (ignored by standard generate)
        essence:          Optional[torch.FloatTensor]    = None,
        epg_node_tokens:  Optional[torch.LongTensor]     = None,
        epg_scalars:      Optional[torch.FloatTensor]    = None,
        epg_categories:   Optional[torch.LongTensor]     = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        # Token embeddings
        x = self.embed(input_ids)

        # EPG prefix — optional, no-op when not provided
        n_prefix = 0
        if epg_node_tokens is not None and epg_scalars is not None and epg_categories is not None:
            prefix   = self.epg_encoder(self.embed(epg_node_tokens), epg_scalars, epg_categories)
            x        = torch.cat([prefix, x], dim=1)
            n_prefix = epg_node_tokens.shape[1]

        # Essence conditioning vector — computed once, injected every N layers
        ess: Optional[torch.Tensor] = None
        if essence is not None:
            ess = self.essence_cond(essence)   # (B, 1, d_model)

        # Backbone
        new_caches: List[Optional[dict]] = []
        for i, layer in enumerate(self.layers):
            lc     = past_key_values[i] if past_key_values is not None else None
            inject = ess if (ess is not None and i % self.config.essence_inject_every_n == 0) else None
            x, nc  = layer(x, essence_cond=inject, cache=lc)
            if use_cache:
                new_caches.append(nc)

        # Slice off EPG prefix before computing logits
        h = x[:, n_prefix:] if n_prefix > 0 else x

        return CausalLMOutputWithPast(
            loss            = None,
            logits          = self.gen_head(h),
            past_key_values = new_caches if use_cache else None,
        )

    # ---- generation helpers -------------------------------------------------

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor,
        past_key_values: Optional[List[dict]] = None, **kwargs,
    ) -> dict:
        # Feed only the new token when KV/SSM cache is populated
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids":       input_ids,
            "past_key_values": past_key_values,
            "use_cache":       kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """Required by beam search — reorder cache along the batch dimension."""
        return [
            {k: v[beam_idx] for k, v in layer_cache.items()}
            if layer_cache else None
            for layer_cache in past_key_values
        ]
