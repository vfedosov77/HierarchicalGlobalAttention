"""Benchmark: OLD ``hga_spda_attention`` vs NEW ``hga_spda_attention_simplified``.

Compares the WHOLE chunk-routed attention MODULE (``CausalSelfAttentionChunkSDPA``) of both
versions end to end -- QKV projection, qk-norm, RoPE, the routing + block-sparse
FlexAttention attend, the xsa/gate mix and the output projection -- NOT just the inner
``chunk_routed_flex_attention`` function. Both versions expose the identical
``forward(x, attn_args, qkvo_w)`` and carry NO parameters of their own (every weight is
passed in via ``qkvo_w`` / ``attn_args``), so the two run on byte-identical inputs.

Conditions (matching the function-level bench):

    ~300K-token fully-packed sequence (T = 307200 -> C = 4800 chunks), 50/100/150 docs with a
    fixed-length padded cu_seqlens (training-loader layout), routing window = 22 chunks,
    topk = 3, H = 6, D = 128, dim = 768, chunk = 64, bf16,
    torch.compile(dynamic=False, fullgraph=True) on the module, each module holding its own
    lazily-built ``_RoutedFlexMaskState`` (persistent BlockMask buffers).

The old version routes the chunk summaries in fp32; the simplified version routes in bf16
(that was the fp32-matmul / TF32-warning fix) and drops every non-default path. Reports fwd
and fwd+bwd wall time + peak memory per version and the speedup, plus a forward numerical
diff between the two module outputs.

Run:  python bench_old_vs_simplified.py
"""
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))          # parent dir holds both modules

from _common import make_packed_batch  # noqa: E402

#import hga_spda_attention as M_OLD                 # restored original (multi-mode) version
import dual_kv_causal_self_attention as M_NEW      # simplified single-mode version

from kernels import get_kernel

flash_attn_interface = get_kernel('kernels-community/flash-attn2', version=1).flash_attn_interface

DEVICE = "cuda"
device=DEVICE
DTYPE = torch.bfloat16
H, D = 6, 128
DIM = H * D                         # model dim (num_heads * head_dim) = 768
CH = 64
TOPK = 3
WINDOW = 14                         # routing window in CHUNKS (bounds the near-diagonal band)
SCALE = 1.0 / (D ** 0.5)

T = 2048 * 8  # *16                     
DOC_COUNTS = [50]
MAX_NUM_DOCS = 256                  # fixed-length padded cu_seqlens (loader layout)
ITERS = 6
WARMUP = 3

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int, num_heads: int, paired: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = dim
        self.hdim = num_heads * head_dim
        self.paired = paired
        assert self.hdim == self.dim, "num_heads * head_dim must equal model_dim"
        # Weights are stored in parameter banks and passed via forward()

    def forward(self, x: Tensor, attn_args: AttnArgs, qkvo_w: Tensor):
        B, T = x.size(0), x.size(1) # batch size, sequence length
        assert B == 1, "varlen sequences requires B == 1"
        assert T % 16 == 0
        # unpack attention args
        aux_v, attn_gate_w = attn_args.aux_v, attn_args.attn_gate_w
        sa_lambdas, key_offset = attn_args.sa_lambdas, attn_args.key_offset
        seqlens, bm_size = attn_args.seqlens, attn_args.bm_size
        train_max_seq_len = WINDOW * 64
        yarn = attn_args.yarn

        q, k, v = F.linear(x, sa_lambdas[0] * qkvo_w[:self.dim * 3].type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        max_len = train_max_seq_len if self.training else (args.val_batch_size // (grad_accum_steps * world_size))

        q, k = norm(q), norm(k) # QK norm @Grad62304977

        if not self.paired:
            q, k = yarn.rotary(q), yarn.rotary(k)

            if key_offset:
                # shift keys forward for the stationary head dims. Enables 1-layer induction.
                k[:, 1:, :, self.head_dim // 2:] = k[:, :-1, :, self.head_dim // 2:]

            if aux_v is not None:
                v = v + aux_v.view_as(v)

        else:
            # Paired heads: adjacent heads' queries attend to each other's keys.
            # Two copies of the input stream are interleaved to achieve this, which:
            # - doubles the length of each sequence
            # - halves the effective window size
            q = q.view(B, T, self.num_heads // 2, self.head_dim * 2)
            k = k.view(B, T, self.num_heads // 2, self.head_dim * 2)
            v = v.reshape(B, T * 2, self.num_heads // 2, self.head_dim)

            q, k = yarn.rotary(q), yarn.rotary(k)

            q = q.view(B, T * 2, self.num_heads // 2, self.head_dim)
            k = k.view(B, T * 2, self.num_heads // 2, self.head_dim)

            if aux_v is not None:
                v = v + aux_v.view_as(v)

            seqlens = 2 * seqlens
            max_len = 2 * max_len

        # use flash_attn over flex_attn @varunneal. flash_attn_varlen suggested by @YouJiacheng
        y = flash_attn_interface.flash_attn_varlen_func(q[0], k[0], v[0], cu_seqlens_q=seqlens, cu_seqlens_k=seqlens,
                                                        max_seqlen_q=max_len, max_seqlen_k=max_len,
                                                        causal=True, softmax_scale=yarn.attn_scale, window_size=(bm_size, 0))
        y = y.view(B, T, self.num_heads, self.head_dim)
        # Gated XSA (arXiv:2603.09078) with learnable strength: subtract per-head fraction tanh(a)
        # of y aligned with v^. Non-paired only (v shape doesn't line up for paired layers).
        if attn_args.xsa_alpha is not None and not self.paired:
            vn = F.normalize(v, dim=-1, eps=1e-4)
            proj = (y * vn).sum(-1, keepdim=True)
            alpha = torch.tanh(attn_args.xsa_alpha).type_as(y).view(1, 1, self.num_heads, 1)
            y = y - alpha * proj * vn
        y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim) # re-assemble all head outputs side by side
        y = F.linear(y, sa_lambdas[1] * qkvo_w[self.dim * 3:].type_as(y))  # sa_lambdas[1] pre-multiplied to O @shenberg
        return y


class Yarn(nn.Module):
    def __init__(self, head_dim, max_seq_len, paired=False):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.paired = paired
        self.reset()

    def rotary(self, x_BTHD):
        assert self.factor1.size(0) >= x_BTHD.size(-3)
        factor1, factor2 = (
            self.factor1[None, : x_BTHD.size(-3), None, :],
            self.factor2[None, : x_BTHD.size(-3), None, :],
        )
        x_flip = x_BTHD.view(*x_BTHD.shape[:-1], x_BTHD.shape[-1] // 2, 2).flip(-1).view(x_BTHD.shape)
        return factor1 * x_BTHD + factor2 * x_flip

    def reset(self):
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=self.head_dim//4, dtype=torch.float32, device=device)
        angular_freq = angular_freq.repeat_interleave(2)
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(self.head_dim//2)])
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=device)
        if not self.paired:
            theta = torch.outer(t, angular_freq)
            self.factor1 = nn.Buffer(
                theta.cos().to(torch.bfloat16), persistent=False
            )
            self.factor2 = nn.Buffer(
                theta.sin().to(torch.bfloat16), persistent=False
            )
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, angular_freq)
            theta2 = torch.outer(t_odd, angular_freq)
            self.factor1 = nn.Buffer(
                torch.cat((theta1.cos(), theta2.cos()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
            self.factor2 = nn.Buffer(
                torch.cat((theta1.sin(), theta2.sin()), dim=-1).to(torch.bfloat16),
                persistent=False
            )
        self.factor2[..., 1::2] *= -1
        self.angular_freq = angular_freq
        # start with 0.1, inspired by 0.12 from @leloykun and learnable scalars used by @brendanh0gan https://x.com/hi_tysam/status/1879693583898591283
        self.attn_scale = 0.1

    def apply(self, old_window: int, new_window: int, alpha: int=1, beta: int=32):
        rotations = old_window * self.angular_freq / (2 * torch.pi)
        scaling_factor = old_window / new_window
        interpolation_weight = torch.clamp((rotations - alpha) / (beta - alpha), 0, 1)
        self.angular_freq *= scaling_factor + interpolation_weight * (1 - scaling_factor)
        t = torch.arange(2*self.max_seq_len, dtype=torch.float32, device=self.angular_freq.device)
        if not self.paired:
            theta = torch.outer(t, self.angular_freq)
            self.factor1.copy_(theta.cos())
            self.factor2.copy_(theta.sin())
        else:
            t_even = 2 * t
            t_odd = t_even + 1
            theta1 = torch.outer(t_even, self.angular_freq)
            theta2 = torch.outer(t_odd, self.angular_freq)
            self.factor1.copy_(torch.cat((theta1.cos(), theta2.cos()), dim=-1))
            self.factor2.copy_(torch.cat((theta1.sin(), theta2.sin()), dim=-1))
        self.factor2[..., 1::2] *= -1
        self.attn_scale *= 0.2 * math.log(new_window / old_window) + 1


def make_module_inputs(seqlens, *, grad, seed):
    """Build the shared module inputs (x, qkvo_w, attn_args). The module holds no params,
    so old & new consume identical tensors. ``grad`` toggles autograd on the fwd+bwd path."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(1, T, DIM, generator=g) * (DIM ** -0.5)).to(DEVICE, DTYPE)
    qkvo_w = (torch.randn(DIM * 6, DIM, generator=g) * (DIM ** -0.5)).to(DEVICE, DTYPE)
    attn_gate_w = (torch.randn(H, 12, generator=g) * (12 ** -0.5)).to(DEVICE, DTYPE)
    sa_lambdas = torch.ones(2, device=DEVICE, dtype=torch.float32)
    yarn = Yarn(D, T)    # factor length 2*ceil(T/2) >= T
    x = x.requires_grad_(grad)
    qkvo_w = qkvo_w.requires_grad_(grad)
    attn_args = SimpleNamespace(
        sa_lambdas=sa_lambdas, seqlens=seqlens, bm_size=0, yarn=yarn, key_offset=False,
        attn_gate_w=attn_gate_w, aux_v=None, xsa_alpha=None, train_max_seq_len=T,
    )
    return x, qkvo_w, attn_args


def make_module(M):
    """Instantiate and compile one version's ``CausalSelfAttentionChunkSDPA`` module."""
    m = M.CausalSelfAttention(
        DIM, D, H, route_window=WINDOW,
    ).to(DEVICE)
    # Register the chunk size so the simplified backward-config patch emits tiles that
    # divide the mask blocks (needed for CH != 64; a no-op for the old module).
    if hasattr(M, "set_flex_bwd_block"):
        M.set_flex_bwd_block(CH)
    return torch.compile(m, dynamic=False, fullgraph=True) #, mode="max-autotune-no-cudagraphs"

def make_module_flash():
    """Instantiate and compile one version's ``CausalSelfAttentionChunkSDPA`` module."""
    m = CausalSelfAttention(
        DIM, D, H, paired=False
    ).to(DEVICE)

    return torch.compile(m, dynamic=False, fullgraph=True) #, mode="max-autotune-no-cudagraphs"


def _bench(mod, x, qkvo_w, attn_args, backward):
    """Return (ms_per_iter, peak_MB) for the whole module, or (None, None) on OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        for _ in range(WARMUP):                                 # warmup also triggers compile
            if backward:
                x.grad = qkvo_w.grad = None
                y = mod(x, attn_args, qkvo_w)
                y.float().pow(2).mean().backward()
            else:
                with torch.no_grad():
                    mod(x, attn_args, qkvo_w)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            if backward:
                x.grad = qkvo_w.grad = None
                y = mod(x, attn_args, qkvo_w)
                y.float().pow(2).mean().backward()
            else:
                with torch.no_grad():
                    mod(x, attn_args, qkvo_w)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / ITERS
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        return ms, peak
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None, None


def _forward_diff(mod_old, mod_new, x, qkvo_w, attn_args):
    """Max/mean abs diff of the two versions' module outputs on the same input."""
    with torch.no_grad():
        y_old = mod_old(x, attn_args, qkvo_w).float()
        y_new = mod_new(x, attn_args, qkvo_w).float()
        d = (y_old - y_new).abs()
        scale = y_old.abs().mean().clamp(min=1e-6)
        return d.max().item(), d.mean().item(), scale.item()


def main():
    C = T // CH
    print(f"device={torch.cuda.get_device_name()}  T={T} C={C} H={H} D={D} dim={DIM} chunk={CH} "
          f"topk={TOPK} window={WINDOW}  docs={DOC_COUNTS}")
    print("module:         CausalSelfAttentionChunkSDPA (full forward: proj + route + attend + gate + out)")
    print("routing dtype:  OLD=fp32 summaries   NEW=bf16 summaries")
    print("compile:        torch.compile(dynamic=False, fullgraph=True) on the module\n")

    #mod_old = make_module(M_OLD)
    mod_new = make_module(M_NEW)
    mod_old = make_module_flash()

    for nd in DOC_COUNTS:
        # make_packed_batch gives the fixed-length padded cu_seqlens (the module's seqlens).
        *_, cu = make_packed_batch(
            T, H, D, n_docs=nd, max_num_docs=MAX_NUM_DOCS, device=DEVICE, dtype=DTYPE, seed=nd,
        )
        header = f"T={T} C={C} docs={nd:4d}"

        for backward in (True,):
            tag = "fwd+bwd" if backward else "fwd    "
            xo, wo, ao = make_module_inputs(cu, grad=backward, seed=nd)
            o_ms, o_pk = _bench(mod_old, xo, wo, ao, backward)
            del xo, wo, ao
            torch.cuda.empty_cache()
            xn, wn, an = make_module_inputs(cu, grad=backward, seed=nd)
         
            n_ms, n_pk = _bench(mod_new, xn, wn, an, backward)
            del xn, wn, an
            
            torch.cuda.empty_cache()

            def fmt(ms, pk):
                return "OOM" if ms is None else f"{ms:8.2f}ms {pk:7.0f}MB"

            if o_ms and n_ms:
                faster = "new" if n_ms < o_ms else "old"
                ratio = max(o_ms, n_ms) / min(o_ms, n_ms)
                verdict = f"  -> {faster} faster ({ratio:.2f}x)"
            else:
                verdict = ""
            print(f"{header} {tag}  old={fmt(o_ms, o_pk)}  "
                  f"new={fmt(n_ms, n_pk)}{verdict}")

        if False:
            xd, wd, ad = make_module_inputs(cu, grad=False, seed=nd)
            dmax, dmean, dscale = _forward_diff(mod_old, mod_new, xd, wd, ad)
            print(f"{header} diff    max={dmax:.3e}  mean={dmean:.3e}  "
                  f"(|old| mean={dscale:.3e})  [routing dtype differs -> a few chunks may reselect]")
            del xd, wd, ad, cu
            torch.cuda.empty_cache()

    


if __name__ == "__main__":
    main()
