"""
Mamba-2 block implementing Structured State Space Duality (SSD).

Core recurrence (maps to whitepaper state equation):
    h_t = A * h_{t-1} + B_t * x_t        (whitepaper: s(t+dt) = alpha * s(t) + beta * g(F,c) + gamma * m)
    y_t = C_t * h_t + D * x_t

The SSD insight: this recurrence with scalar-structured A is equivalent to a form
of structured masked attention, enabling chunked computation on matrix multiply hardware.
"""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Mamba2Config:
    d_model: int = 1024
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    n_heads: int = 16
    norm_eps: float = 1e-5
    chunk_size: int = 64  # SSD chunk size for parallel scan


class Mamba2Block(nn.Module):
    """
    Single Mamba-2 layer with selective state space duality.

    Architecture per block:
        Input (d_model)
            -> in_proj -> (z, x, B, C, dt)
            -> conv1d(x)
            -> selective_scan(x, A, B, C, dt) -> y
            -> y * silu(z)  (output gating)
            -> out_proj -> Output (d_model)
    """

    def __init__(self, cfg: Mamba2Config):
        super().__init__()
        self.cfg = cfg
        self.d_inner = cfg.d_model * cfg.expand
        self.head_dim = self.d_inner // cfg.n_heads

        assert self.d_inner % cfg.n_heads == 0, (
            f"d_inner ({self.d_inner}) must be divisible by n_heads ({cfg.n_heads})"
        )

        # Input projection: d_model -> (z, x, B, C, dt)
        # z: gating branch (d_inner)
        # x: SSM input branch (d_inner)
        # B: input matrix per head (n_heads * d_state)
        # C: output matrix per head (n_heads * d_state)
        # dt: time step per head (n_heads)
        self.in_proj_dim = (
            self.d_inner  # z
            + self.d_inner  # x
            + cfg.n_heads * cfg.d_state  # B
            + cfg.n_heads * cfg.d_state  # C
            + cfg.n_heads  # dt
        )
        self.in_proj = nn.Linear(cfg.d_model, self.in_proj_dim, bias=False)

        self.conv1d_weight = mx.zeros((self.d_inner, cfg.d_conv))
        self.conv1d_bias = mx.zeros((self.d_inner,))

        # A parameter (log space, one scalar per head — SSD constraint)
        # Initialized negative so exp(A_log) gives decay < 1
        self.A_log = mx.full((cfg.n_heads,), math.log(0.5))

        self.D = mx.ones((cfg.n_heads,))

        self.dt_bias = mx.zeros((cfg.n_heads,))

        self.out_proj = nn.Linear(self.d_inner, cfg.d_model, bias=False)

        self.norm = nn.RMSNorm(self.d_inner, eps=cfg.norm_eps)

    def _causal_conv1d(self, x: mx.array, cache: mx.array | None = None) -> tuple[mx.array, mx.array]:
        """
        Causal 1D convolution over the sequence dimension.

        Args:
            x: (batch, seq_len, d_inner)
            cache: (batch, d_conv - 1, d_inner) — conv state from previous step

        Returns:
            y: (batch, seq_len, d_inner)
            new_cache: (batch, d_conv - 1, d_inner)
        """
        B, L, D = x.shape
        k = self.cfg.d_conv

        if cache is not None:
            # Prepend cached context for causal continuity
            x_padded = mx.concatenate([cache, x], axis=1)
        else:
            x_padded = mx.concatenate([mx.zeros((B, k - 1, D)), x], axis=1)

        # Depthwise conv1d via loop over kernel width
        # MLX is lazy-evaluated so this compiles efficiently
        y = mx.zeros((B, L, D))
        for i in range(k):
            y = y + x_padded[:, i : i + L, :] * self.conv1d_weight[:, i]

        y = y + self.conv1d_bias

        # Cache: last (k-1) positions of x for next call
        new_cache = x_padded[:, -(k - 1) :, :]

        return y, new_cache

    def _selective_scan(
        self,
        x: mx.array,
        A: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        D: mx.array,
        ssm_state: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Selective scan — the core SSM recurrence.

        This implements the whitepaper's state update:
            h_t = exp(A * dt_t) * h_{t-1} + dt_t * B_t * x_t
            y_t = C_t * h_t + D * x_t

        Uses sequential scan. For training, the SSD chunked parallel form
        should be used instead (see _ssd_chunked).

        Args:
            x:  (batch, seq_len, n_heads, head_dim)
            A:  (n_heads,) — log-space decay rates
            B:  (batch, seq_len, n_heads, d_state)
            C:  (batch, seq_len, n_heads, d_state)
            dt: (batch, seq_len, n_heads)
            D:  (n_heads,) — skip connection
            ssm_state: (batch, n_heads, head_dim, d_state) — recurrent state

        Returns:
            y: (batch, seq_len, n_heads, head_dim)
            final_state: (batch, n_heads, head_dim, d_state)
        """
        batch, seq_len, n_heads, head_dim = x.shape
        d_state = B.shape[-1]

        # Discretize A: dA = exp(A * dt)
        # A is (n_heads,), dt is (batch, seq_len, n_heads)
        # dt[:, :, :, None] is (batch, seq_len, n_heads, 1) — broadcasts naturally
        dA = mx.exp(A[None, None, :, None] * dt[:, :, :, None])  # (batch, seq_len, n_heads, 1)

        # Discretize B: dB = dt * B
        dB = dt[:, :, :, None] * B  # (batch, seq_len, n_heads, d_state)

        if ssm_state is None:
            h = mx.zeros((batch, n_heads, head_dim, d_state))
        else:
            h = ssm_state

        ys = []
        for t in range(seq_len):
            # State update: h = dA * h + dB * x
            # dA[:, t]: (batch, n_heads, 1) — broadcasts over (head_dim, d_state)
            # dB[:, t]: (batch, n_heads, d_state) — needs (batch, n_heads, 1, d_state)
            # x[:, t]:  (batch, n_heads, head_dim) — needs (batch, n_heads, head_dim, 1)
            dA_t = dA[:, t, :, :][:, :, None, :]   # (batch, n_heads, 1, 1)
            dB_t = dB[:, t, :, :][:, :, None, :]   # (batch, n_heads, 1, d_state)
            x_t = x[:, t, :, :, None]              # (batch, n_heads, head_dim, 1)
            h = dA_t * h + dB_t * x_t

            # Output: y = C * h + D * x
            # C[:, t]: (batch, n_heads, d_state) → (batch, n_heads, 1, d_state)
            C_t = C[:, t, :, :][:, :, None, :]     # (batch, n_heads, 1, d_state)
            y_t = (C_t * h).sum(axis=-1)            # (batch, n_heads, head_dim)
            y_t = y_t + D[None, :, None] * x[:, t, :, :]

            ys.append(y_t)

        y = mx.stack(ys, axis=1)  # (batch, seq_len, n_heads, head_dim)
        return y, h

    def _ssd_chunked(
        self,
        x: mx.array,
        A: mx.array,
        B: mx.array,
        C: mx.array,
        dt: mx.array,
        D: mx.array,
        chunk_size: int | None = None,
    ) -> mx.array:
        """
        SSD chunked parallel computation for training.

        Splits the sequence into chunks and processes each chunk as a
        matrix multiplication (leveraging the SSD duality). Inter-chunk
        dependencies are handled by passing state between chunks.

        This is mathematically equivalent to _selective_scan but parallelizes
        better on matrix multiply hardware (Metal tensor cores).
        """
        batch, seq_len, n_heads, head_dim = x.shape
        d_state = B.shape[-1]
        cs = chunk_size or self.cfg.chunk_size

        pad_len = (cs - seq_len % cs) % cs
        if pad_len > 0:
            x = mx.concatenate([x, mx.zeros((batch, pad_len, n_heads, head_dim))], axis=1)
            B = mx.concatenate([B, mx.zeros((batch, pad_len, n_heads, d_state))], axis=1)
            C = mx.concatenate([C, mx.zeros((batch, pad_len, n_heads, d_state))], axis=1)
            dt = mx.concatenate([dt, mx.zeros((batch, pad_len, n_heads))], axis=1)

        n_chunks = (seq_len + pad_len) // cs

        x_c = x.reshape(batch, n_chunks, cs, n_heads, head_dim)
        B_c = B.reshape(batch, n_chunks, cs, n_heads, d_state)
        C_c = C.reshape(batch, n_chunks, cs, n_heads, d_state)
        dt_c = dt.reshape(batch, n_chunks, cs, n_heads)

        # Compute cumulative decay within each chunk
        # dA_cum[i] = exp(A * sum(dt[0:i]))
        dt_cum = mx.cumsum(dt_c, axis=2)  # (batch, n_chunks, cs, n_heads)
        dA_cum = mx.exp(A[None, None, None, :] * dt_cum)

        # Intra-chunk: compute attention-like matrix L
        # L[i,j] = C[i] * (prod_{k=j+1}^{i} dA[k]) * B[j] for j <= i, else 0
        # This is the "dual form" — SSM as structured masked attention

        # For each chunk, compute the (cs x cs) lower-triangular attention matrix
        # and use it to compute output via matmul
        outputs = []
        h = mx.zeros((batch, n_heads, head_dim, d_state))

        for chunk_idx in range(n_chunks):
            xc = x_c[:, chunk_idx]    # (batch, cs, n_heads, head_dim)
            Bc = B_c[:, chunk_idx]    # (batch, cs, n_heads, d_state)
            Cc = C_c[:, chunk_idx]    # (batch, cs, n_heads, d_state)
            dtc = dt_c[:, chunk_idx]  # (batch, cs, n_heads)

            # Compute decay factors within chunk
            dt_cum_c = mx.cumsum(dtc, axis=1)  # (batch, cs, n_heads)

            # For positions i, j in chunk: relative decay = exp(A * (cum_dt[i] - cum_dt[j]))
            # Shape: (batch, cs, cs, n_heads)
            decay_matrix = mx.exp(
                A[None, None, None, :]
                * (dt_cum_c[:, :, None, :] - dt_cum_c[:, None, :, :])
            )

            causal_mask = mx.tri(cs, cs, k=0)  # (cs, cs) lower triangular
            decay_matrix = decay_matrix * causal_mask[None, :, :, None]

            # Intra-chunk attention: L[b,i,j,h] * (B[b,j,h,:] @ C[b,i,h,:])
            # Compute B @ C^T per head: (batch, cs, cs, n_heads)
            # BC[b, i, j, h] = sum_s C[b,i,h,s] * B[b,j,h,s]
            BC = mx.sum(
                Cc[:, :, None, :, :] * Bc[:, None, :, :, :],
                axis=-1,
            )  # (batch, cs, cs, n_heads)

            attn = decay_matrix * BC  # (batch, cs, cs, n_heads)

            # Apply to x: y_intra[b,i,h,d] = sum_j attn[b,i,j,h] * x[b,j,h,d]
            # attn: (batch, cs, cs, n_heads) -> (batch, cs, cs, n_heads, 1)
            # xc: (batch, cs, n_heads, head_dim) -> (batch, 1, cs, n_heads, head_dim)
            y_intra = mx.sum(
                attn[:, :, :, :, None] * xc[:, None, :, :, :],
                axis=2,
            )  # (batch, cs, n_heads, head_dim)

            # Inter-chunk: contribution from previous state h
            # decay from chunk start to each position
            decay_from_start = mx.exp(
                A[None, None, :] * dt_cum_c
            )  # (batch, cs, n_heads)

            # y_inter[b,i,h,d] = sum_s C[b,i,h,s] * (decay[i] * h[b,h,d,s])
            # h: (batch, n_heads, head_dim, d_state)
            h_decayed = decay_from_start[:, :, :, None, None] * h[:, None, :, :, :]
            # h_decayed: (batch, cs, n_heads, head_dim, d_state)
            y_inter = mx.sum(
                Cc[:, :, :, None, :] * h_decayed,
                axis=-1,
            )  # (batch, cs, n_heads, head_dim)

            y_chunk = y_intra + y_inter

            y_chunk = y_chunk + D[None, None, :, None] * xc

            outputs.append(y_chunk)

            # Update state for next chunk
            # h_new = dA_total * h + sum_t (dA_from_t * dB_t * x_t)
            total_decay = mx.exp(
                A[None, :] * dt_cum_c[:, -1, :]
            )  # (batch, n_heads)
            h = total_decay[:, :, None, None] * h

            for t in range(cs):
                decay_from_t = mx.exp(
                    A[None, :] * (dt_cum_c[:, -1, :] - dt_cum_c[:, t, :])
                )  # (batch, n_heads)
                dB_t = dtc[:, t, :, None] * Bc[:, t]  # (batch, n_heads, d_state)
                h = h + (
                    decay_from_t[:, :, None, None]
                    * dB_t[:, :, None, :]
                    * xc[:, t, :, :, None]
                )

        y = mx.concatenate(outputs, axis=1)  # (batch, seq_len + pad, n_heads, head_dim)

        if pad_len > 0:
            y = y[:, :seq_len]

        return y

    def __call__(
        self,
        x: mx.array,
        cache: dict | None = None,
        use_chunked: bool = True,
    ) -> tuple[mx.array, dict | None]:
        """
        Forward pass.

        Args:
            x: (batch, seq_len, d_model)
            cache: optional recurrent state dict for inference
            use_chunked: if True, use SSD parallel form (training);
                         if False, use sequential scan (inference)

        Returns:
            output: (batch, seq_len, d_model)
            new_cache: updated cache dict (or None if no cache)
        """
        batch, seq_len, _ = x.shape
        cfg = self.cfg

        proj = self.in_proj(x)  # (batch, seq_len, in_proj_dim)

        # Split projections
        idx = 0
        z = proj[:, :, idx : idx + self.d_inner]
        idx += self.d_inner
        x_ssm = proj[:, :, idx : idx + self.d_inner]
        idx += self.d_inner
        B = proj[:, :, idx : idx + cfg.n_heads * cfg.d_state]
        idx += cfg.n_heads * cfg.d_state
        C = proj[:, :, idx : idx + cfg.n_heads * cfg.d_state]
        idx += cfg.n_heads * cfg.d_state
        dt_raw = proj[:, :, idx : idx + cfg.n_heads]

        # Reshape for multi-head
        B = B.reshape(batch, seq_len, cfg.n_heads, cfg.d_state)
        C = C.reshape(batch, seq_len, cfg.n_heads, cfg.d_state)

        # Softplus on dt to ensure positive, then add bias
        dt = nn.softplus(dt_raw) + self.dt_bias[None, None, :]  # (batch, seq_len, n_heads)

        # Causal conv1d on x
        conv_cache = cache.get("conv") if cache else None
        x_ssm, new_conv_cache = self._causal_conv1d(x_ssm, conv_cache)
        x_ssm = nn.silu(x_ssm)

        # Reshape x for multi-head SSM
        x_ssm = x_ssm.reshape(batch, seq_len, cfg.n_heads, self.head_dim)

        # Get A from log space (negative for decay)
        A = -mx.exp(self.A_log)  # (n_heads,) — negative values

        # Run SSM
        if use_chunked and seq_len > cfg.chunk_size:
            y = self._ssd_chunked(x_ssm, A, B, C, dt, self.D)
            new_ssm_state = None
        else:
            ssm_state = cache.get("ssm") if cache else None
            y, new_ssm_state = self._selective_scan(x_ssm, A, B, C, dt, self.D, ssm_state)

        # Reshape back: (batch, seq_len, n_heads, head_dim) -> (batch, seq_len, d_inner)
        y = y.reshape(batch, seq_len, self.d_inner)

        # Normalize, then gate with z
        y = self.norm(y)
        y = y * nn.silu(z)

        output = self.out_proj(y)

        # Build cache
        new_cache = None
        if cache is not None:
            new_cache = {"conv": new_conv_cache, "ssm": new_ssm_state}

        return output, new_cache
