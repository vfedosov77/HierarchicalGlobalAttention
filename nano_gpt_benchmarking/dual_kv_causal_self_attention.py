# dual_kv_causal_self_attention_v9_strict_prepare.py
# Drop-in CausalSelfAttention replacement using one FlexAttention call over
# [short K/V table | long K/V table].
#
# v9 (addresses real-world torch.compile(fullgraph=True) + variable/packed lengths):
# - REMOVED the automatic prepare inside forward(). That code, while convenient,
#   introduced Python control flow + potential exceptions that Dynamo treats as
#   "Observed exception" ? graph break / Unsupported when lengths mismatch.
# - STRICT contract for fullgraph compilation:
#     You MUST call prepare_mask_cache(exact_T, device) **before** torch.compile
#     and before any forward() that will run under that compiled graph, for the
#     *actual* T = x.size(1) that will be seen at runtime.
#   This guarantees that the BlockMask's seq_lengths exactly matches the q/kv
#   tensors passed to flex_attention ? no ValueError from FlexAttention inside
#   the compiled region.
# - Still fully supports *multiple different sequence lengths* via the
#   _mask_states cache. You can prepare several common Ts before compile;
#   Dynamo will specialize one graph per distinct T (or use mark_dynamic).
# - Eager mode (no compile) is now more forgiving: if lengths differ we
#   automatically try block_mask._adjust(q_len, kv_len) before falling back
#   to raising (the adjust may or may not be semantically correct for every
#   custom mask_mod  test it).
# - All previous v7/v8 guarantees preserved: fullgraph-friendly hot path,
#   no raises from *our* code inside forward when properly prepared,
#   compact O(C*W) BlockMask, RMS qk-norm, etc.
#
# Typical correct usage with torch.compile(fullgraph=True):
#     attn = CausalSelfAttention(..., static_seq_len=None, ...)
#     # Determine the T your training batches will actually have
#     example_T = 8192   # or inputs.shape[1] from a real batch
#     attn.prepare_mask_cache(example_T, device)
#     model = torch.compile(model, fullgraph=True, ...)
#     # later, if you change to a different common length:
#     attn.prepare_mask_cache(another_T, device)  # recompile will happen once
#
# If you truly have many different Ts or packed sequences with cum_seqlens,
# consider either (a) padding everything to a fixed max T, or (b) moving
# mask creation outside the compiled region and passing the mask explicitly.
from __future__ import annotations
import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask as _create_block_mask,
    flex_attention as _torch_flex_attention,
)


def _patch_flex_bwd_configs():
    try:
        from torch._inductor.template_heuristics.triton import (
            CUDAConfigHeuristic,
            FlexBwDConfig,
        )
    except ImportError:
        return
    if getattr(CUDAConfigHeuristic.get_flex_attn_bwd_configs, "_hga_patched", False):
        return
    orig = CUDAConfigHeuristic.get_flex_attn_bwd_configs
    single = FlexBwDConfig(64, 64, 64, 64, 2, 4)
    def patched(self, head_dim, dtype):
        if head_dim == 128:
            return [single]
        cfgs = list(orig(self, head_dim, dtype))
        if single not in cfgs:
            cfgs.append(single)
        return cfgs
    patched._hga_patched = True
    CUDAConfigHeuristic.get_flex_attn_bwd_configs = patched


_patch_flex_bwd_configs()


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


# FlexAttention only emits the fused, block-mask-aware O(C*W) kernel when compiled.
# In eager it materializes the full [T, 2T] scores matrix, so we compile the call
# lazily (once) and reuse it. If we are already inside a compiled graph, use the
# raw callable so it gets inlined instead of nested-compiled.
_flex_attention_compiled = None


def _get_compiled_flex():
    global _flex_attention_compiled
    if _flex_attention_compiled is None:
        _flex_attention_compiled = torch.compile(_torch_flex_attention, dynamic=False)
    return _flex_attention_compiled


class _DualKVCompactWindowMaskState:
    """
    Blocked dual-KV layout: KV columns [0, T) are short-term keys and
    [T, 2T) are long-term keys (same token order). A query at position q
    (chunk c = q // CH) attends:

      short key at token t (chunk k):  causal (q >= t) AND (c - k) <= Ws
      long  key at token t (chunk k):  causal (q >= t) AND (c - k) <= Wl

    Ws = short_window, Wl = long_window (in chunks). Ws/Wl == C-1 opens the
    whole (causal) context for that branch.

    The BlockMask is built with the official ``create_block_mask`` helper so
    the sparse block metadata is exactly what the compiled flex_attention
    kernel expects (hand-rolling the metadata is fragile and was silently
    misread by the Triton kernel).
    """
    def __init__(self, C: int, T: int, chunk_size: int, short_window: int, long_window: int, device):
        self.C = int(C)
        self.T = int(T)
        self.CH = int(chunk_size)
        self.short_window = max(0, min(int(short_window), self.C - 1))
        self.long_window = max(0, min(int(long_window), self.C - 1))
        self.device = device
        T = self.T
        CH = self.CH
        Ws = self.short_window
        Wl = self.long_window

        def dual_window_mask(b, h, q_idx, kv_idx):
            is_long = kv_idx >= T
            tok = torch.where(is_long, kv_idx - T, kv_idx)
            causal = q_idx >= tok
            chunk_gap = (q_idx // CH) - (tok // CH)
            within = torch.where(is_long, chunk_gap <= Wl, chunk_gap <= Ws)
            return causal & within

        self.mask = _create_block_mask(
            dual_window_mask,
            B=None,
            H=None,
            Q_LEN=T,
            KV_LEN=2 * T,
            device=device,
            BLOCK_SIZE=CH,
        )


def _get_dual_kv_compact_window_mask_state(
    cache: dict,
    C: int,
    T: int,
    chunk_size: int,
    short_window: int,
    long_window: int,
    device,
) -> _DualKVCompactWindowMaskState:
    key = (int(C), int(T), int(chunk_size), int(short_window), int(long_window), str(device))
    st = cache.get(key)
    if st is None:
        st = _DualKVCompactWindowMaskState(C, T, chunk_size, short_window, long_window, device)
        cache[key] = st
    return st


def _dual_kv_flex_attention(
    q: Tensor,
    k_short: Tensor,
    v_short: Tensor,
    k_long: Tensor,
    v_long: Tensor,
    decay_rate: Tensor,
    chunk_size: int,
    softmax_scale: float,
    block_mask: BlockMask,
    gate_floor: float,
) -> Tensor:
    seq_len, _, _ = q.shape          # actual runtime sequence length
    CH = int(chunk_size)
    k = torch.cat((k_short, k_long), dim=0)
    v = torch.cat((v_short, v_long), dim=0)
    qf = q.transpose(0, 1)[None]
    kf = k.transpose(0, 1)[None]
    vf = v.transpose(0, 1)[None]
    floor = float(gate_floor)
    one_minus_2floor = 1.0 - 2.0 * floor

    # In eager mode we can try to auto-adjust a mismatched BlockMask.
    # This is *not* done under torch.compile (would be a graph break + the
    # adjust is not guaranteed to be correct for every mask_mod).
    # Under compile you must have prepared the mask for the exact seq_len.
    if not torch.compiler.is_compiling():
        expected_q = block_mask.seq_lengths[0]
        expected_kv = block_mask.seq_lengths[1]
        if (expected_q != seq_len) or (expected_kv != 2 * seq_len):
            try:
                block_mask = block_mask._adjust(seq_len, 2 * seq_len)
            except Exception:
                # If adjust fails or is not appropriate for this mask_mod,
                # we let the normal flex_attention error happen below.
                pass

    def score_mod(score, b, h, q_idx, kv_idx):
        is_long = kv_idx >= seq_len
        kv_token_idx = torch.where(is_long, kv_idx - seq_len, kv_idx)
        dist = (q_idx - kv_token_idx).to(torch.float32).clamp_min(0.0)
        rate = decay_rate[b, h, kv_token_idx]
        x = rate * dist
        if floor > 0.0:
            p_short = floor + one_minus_2floor * torch.exp(-x)
            p_short = torch.clamp(p_short, min=1.0e-6, max=1.0 - 1.0e-6)
            short_bias = torch.log(p_short)
            long_bias = torch.log1p(-p_short)
        else:
            short_bias = -x
            x_safe = torch.clamp(x, min=1.0e-6)
            long_bias = torch.log1p(-torch.exp(-x_safe))
            long_bias = torch.where(
                dist > 0,
                long_bias,
                torch.full_like(long_bias, -24.0),
            )
        branch_bias = torch.where(is_long, long_bias, short_bias)
        return score + branch_bias.to(score.dtype)

    flex = _torch_flex_attention if torch.compiler.is_compiling() else _get_compiled_flex()
    return flex(
        qf,
        kf,
        vf,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=float(softmax_scale),
        kernel_options={
            "fwd_BLOCK_M": CH,
            "fwd_BLOCK_N": CH,
            "fwd_num_warps": 4,
            "fwd_num_stages": 3,
            "ROWS_GUARANTEED_SAFE": True,
        },
    )[0].transpose(0, 1)


class CausalSelfAttention(nn.Module):
    """
    qkvo_w layout:
        qkvo_w[:5*dim] -> Q, K_short, V_short, K_long, V_long
        qkvo_w[5*dim:] -> output projection
    route_window is in chunks. For ~1300 tokens with route_chunk_size=64,
    pass route_window=20 or 21, not 1300.

    === CRITICAL CONTRACT FOR torch.compile(fullgraph=True) ===
    The BlockMask is *static* and its seq_lengths must exactly match the
    q/kv tensor lengths that will be passed to flex_attention at runtime.
    Therefore you MUST:

        attn.prepare_mask_cache(actual_T, device)   # actual_T = x.size(1) in forward

    BEFORE you do torch.compile(..., fullgraph=True) and before any forward
    call that will run under the compiled graph.

    If you prepared for T=192 but your real batches have T=8192 (or vice-versa),
    FlexAttention will raise ValueError inside the compiled region.
    Dynamo then turns that into:
        torch._dynamo.exc.Unsupported: Observed exception

    This is why v9 removed the "lazy prepare inside forward" that existed in v8.
    That lazy code could itself trigger the observed-exception path.

    Correct pattern:
        # once (or when length changes)
        for layer in model.layers:
            layer.attn.prepare_mask_cache(8192, device)   # or your real T
        model = torch.compile(model, fullgraph=True, ...)

    The class still caches every distinct (T, device, window) you prepare,
    so switching between a few common lengths is cheap after the first time.
    """

    def __init__(
        self,
        dim: int,
        head_dim: int,
        num_heads: int,
        static_seq_len: int,
        static_device,
        route_window: int = 0,
        route_chunk_size: int = 64,
        *,
        min_half_life: float = 1.0,
        max_half_life: float = 4096.0,
        init_half_life: float = 64.0,
        init_half_life_spread: float = 8.0,
        dual_kv_gate_floor: float = 0.0,
        short_route_window: int | None = 2,
        qk_norm_fn: Callable[[Tensor], Tensor] | None = None,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dim = int(dim)
        self.hdim = self.num_heads * self.head_dim
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        self.route_chunk_size = int(route_chunk_size)
        self.route_window = int(route_window)
        self._mask_states: dict = {}
        self._prepared_block_mask = None
        self._prepared_mask_t = -1
        self._prepared_mask_device = None
        self._prepared_mask_window = -1
        self.qk_norm_fn = qk_norm_fn
        self.static_seq_len = int(static_seq_len) if static_seq_len is not None else -1
        self.min_half_life = float(min_half_life)
        self.max_half_life = float(max_half_life)
        self.init_half_life = float(init_half_life)
        self.init_half_life_spread = float(init_half_life_spread)
        self.dual_kv_gate_floor = float(dual_kv_gate_floor)
        # Short-branch routing window, in chunks. None -> whole sequence (like long).
        self.short_route_window = short_route_window
        self.halflife_proj = nn.Linear(self.dim, self.num_heads, bias=True)
        self._init_halflife_projection()
        if static_seq_len is not None and static_device is not None:
            self.prepare_mask_cache(static_seq_len, static_device)

    def _apply(self, fn):
        module = super()._apply(fn)
        if self.static_seq_len > 0:
            try:
                device = next(self.parameters()).device
                self.prepare_mask_cache(self.static_seq_len, device)
            except StopIteration:
                pass
        return module

    def get_block_mask(self, T: int, device, route_window: int | None = None) -> BlockMask:
        """
        Return the compact dual-KV BlockMask for this exact (T, route_window),
        building it once and caching it. THIS is the mask cache: every distinct
        (C, T, chunk_size, W, device) is memoized in self._mask_states, so after the
        first time a given combination is seen this is just an O(1) dict lookup and
        the BlockMask tensors are reused (no rebuild, and no torch.compile recompile
        because the same object/shapes are passed to the compiled flex call).

        route_window is in chunks; if None the module-level self.route_window is used.
        T must be divisible by route_chunk_size.
        """
        T = int(T)
        if T % self.route_chunk_size != 0:
            raise ValueError(f"T={T} must be divisible by route_chunk_size={self.route_chunk_size}.")
        device = torch.device(device)
        C = T // self.route_chunk_size
        rw = self.route_window if route_window is None else int(route_window)
        long_window = max(0, min(rw, C - 1))
        if self.short_route_window is None:
            short_window = max(0, C - 1)
        else:
            short_window = max(0, min(int(self.short_route_window), C - 1))
        mask_state = _get_dual_kv_compact_window_mask_state(
            self._mask_states, C, T, self.route_chunk_size, short_window, long_window, device
        )
        return mask_state.mask

    def prepare_mask_cache(self, T: int, device, route_window: int | None = None) -> None:
        """
        Prime the mask cache for (T, route_window) and remember it as the "last
        prepared" mask. Thin wrapper over get_block_mask() kept for API compatibility;
        the actual caching lives in get_block_mask()/self._mask_states.

        T must be divisible by route_chunk_size.
        """
        rw = self.route_window if route_window is None else int(route_window)
        self._prepared_block_mask = self.get_block_mask(T, device, rw)
        self._prepared_mask_t = int(T)
        self._prepared_mask_device = torch.device(device)
        self._prepared_mask_window = rw

    def forward(self, x: Tensor, attn_args, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1)

        # Derive the routing window (in chunks) at runtime from the per-layer
        # sliding-window size `bm_size` (in tokens), then fetch the matching compact
        # BlockMask straight from the mask cache (self._mask_states via get_block_mask).
        # The first time a given (T, route_window) is seen the mask is built and cached;
        # every subsequent call is an O(1) dict lookup. forward() runs in eager Python
        # here (only the flex_attention call itself is torch.compiled), so building /
        # looking up the mask inside forward is safe.
        bm_size = getattr(attn_args, "bm_size", None)
        if bm_size is None:
            route_window = self.route_window
        else:
            # ceil division: window must cover at least bm_size tokens.
            route_window = (int(bm_size) + self.route_chunk_size - 1) // self.route_chunk_size
        block_mask = self.get_block_mask(T, x.device, route_window)

        aux_v = attn_args.aux_v
        attn_gate_w = attn_args.attn_gate_w
        sa_lambdas = attn_args.sa_lambdas
        key_offset = attn_args.key_offset
        yarn = attn_args.yarn

        q, k_short, v_short, k_long, v_long = F.linear(
            x,
            sa_lambdas[0] * qkvo_w[: self.dim * 5].type_as(x),
        ).view(B, T, 5 * self.num_heads, self.head_dim).chunk(5, dim=-2)

        q = norm(q)
        k_short = norm(k_short)
        k_long = norm(k_long)

        q = yarn.rotary(q)
        k_short = yarn.rotary(k_short)
        k_long = yarn.rotary(k_long)

        if key_offset:
            k_short[:, 1:, :, self.head_dim // 2:] = k_short[:, :-1, :, self.head_dim // 2:]
            k_long[:, 1:, :, self.head_dim // 2:] = k_long[:, :-1, :, self.head_dim // 2:]

        if aux_v is not None:
            aux = aux_v.view_as(v_short)
            v_short = v_short + aux
            v_long = v_long + aux

        # Optional per-layer halflife projection injected by the caller (parameter bank).
        halflife_w = getattr(attn_args, "halflife_w", None)
        halflife_b = getattr(attn_args, "halflife_b", None)
        decay_rate = self._key_decay_rate(x, weight=halflife_w, bias=halflife_b)

        y = _dual_kv_flex_attention(
            q[0],
            k_short[0],
            v_short[0],
            k_long[0],
            v_long[0],
            decay_rate,
            chunk_size=self.route_chunk_size,
            softmax_scale=yarn.attn_scale,
            block_mask=block_mask,
            gate_floor=self.dual_kv_gate_floor,
        ).unsqueeze(0)

        if attn_args.xsa_alpha is not None:
            vn = F.normalize(v_short, dim=-1, eps=1e-4)
            proj = (y * vn).sum(-1, keepdim=True)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)
            y = y - alpha * proj * vn

        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w.type_as(x))).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.hdim)
        return F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 5:].type_as(y))

    def _key_decay_rate(self, x: Tensor, weight: Tensor | None = None, bias: Tensor | None = None) -> Tensor:
        if weight is None:
            weight = self.halflife_proj.weight
            bias = self.halflife_proj.bias
        z = F.linear(
            x,
            weight.type_as(x),
            bias.type_as(x),
        ).float()
        log_min = math.log(self.min_half_life)
        log_max = math.log(self.max_half_life)
        log_half_life = log_min + (log_max - log_min) * torch.sigmoid(z)
        decay_rate = math.log(2.0) * torch.exp(-log_half_life)
        return decay_rate.transpose(1, 2).contiguous()

    def _initial_half_lives(self, n: int, device=None) -> Tensor:
        center = min(max(self.init_half_life, self.min_half_life), self.max_half_life)
        if n == 1 or self.init_half_life_spread == 1.0:
            return torch.full((n,), center, dtype=torch.float32, device=device)
        lo = max(self.min_half_life, center / self.init_half_life_spread)
        hi = min(self.max_half_life, center * self.init_half_life_spread)
        if lo >= hi:
            return torch.full((n,), center, dtype=torch.float32, device=device)
        return torch.exp(torch.linspace(math.log(lo), math.log(hi), n, dtype=torch.float32, device=device))

    def _init_halflife_projection(self):
        nn.init.zeros_(self.halflife_proj.weight)
        half_lives = self._initial_half_lives(self.num_heads, device=self.halflife_proj.bias.device)
        log_min = math.log(self.min_half_life)
        log_max = math.log(self.max_half_life)
        p = (half_lives.log() - log_min) / (log_max - log_min)
        p = torch.clamp(p, min=1e-4, max=1.0 - 1e-4)
        with torch.no_grad():
            self.halflife_proj.bias.copy_(torch.log(p / (1.0 - p)))

    @torch.no_grad()
    def current_half_life(self, x: Tensor) -> Tensor:
        z = F.linear(
            x,
            self.halflife_proj.weight.type_as(x),
            self.halflife_proj.bias.type_as(x),
        ).float()
        log_min = math.log(self.min_half_life)
        log_max = math.log(self.max_half_life)
        log_half_life = log_min + (log_max - log_min) * torch.sigmoid(z)
        return log_half_life.exp().transpose(1, 2).contiguous()
