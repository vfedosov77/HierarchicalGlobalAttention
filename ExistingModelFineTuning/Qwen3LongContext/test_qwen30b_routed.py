#!/usr/bin/env python3
"""Quality test: KvRouter-based sparse attention vs. original Qwen3-30B-A3B-FP8.

Two stages:

1. ``selftest_exact_equivalence`` (fast, no model) — proves the exact router's assembly/masking
   is correct: with ``keep_last`` >= all chunks and no routed middle, ``decode_block_exact``
   streamed chunk-by-chunk must reproduce dense causal attention (``F.scaled_dot_product_attention``).

2. ``compare_on_qwen`` — loads the FP8 30B model once and runs a single teacher-forced forward over
   a ~2000-token context under three attention implementations that share the *same* projections,
   RoPE and router config:
       * baseline : original Qwen3 dense attention (the reference)
       * summary  : stock ChunkRouter (group-summary K/V) == HierarchicalGlobalAttentionRouted logic
       * exact    : the new ExactChunkRouter (real token KV of selected chunks)
   Reports greedy next-token agreement vs. baseline (overall + on the routing-active tail) and
   per-token perplexity for each, plus peak VRAM.

Run (venv ~/my_env):
    python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed
    python -m ExistingModelFineTuning.Qwen3LongContext.test_qwen30b_routed --selftest-only
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention import (
    ExactChunkRouter,
    QwenRoutedAttention,
    replace_qwen_attention_with_router,
    restore_original_attention,
)
from ExistingModelFineTuning.KvRouter import RouterConfig, VramKVCacheStore
from ExistingModelFineTuning.KvRouter.cache_store import ChunkPlacementPolicy


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"


def gb(x: int) -> float:
    return x / 1024**3


# -------------------------------------------------------------------------------------------------
# Stage 1: exact-router == dense causal attention
# -------------------------------------------------------------------------------------------------
def selftest_exact_equivalence(device: str = "cpu") -> None:
    torch.manual_seed(0)
    B, H, KVH, Dh = 1, 4, 2, 16
    C, gs, S = 8, 4, 37
    rep = H // KVH

    q = torch.randn(B, H, S, Dh, device=device)
    k = torch.randn(B, KVH, S, Dh, device=device)
    v = torch.randn(B, KVH, S, Dh, device=device)

    # Reference: dense causal attention with GQA expansion.
    k_h = k.repeat_interleave(rep, dim=1)
    v_h = v.repeat_interleave(rep, dim=1)
    ref = F.scaled_dot_product_attention(q, k_h, v_h, is_causal=True)

    # Routed exact path with everything kept local (keep_last covers all chunks, no middle).
    cfg = RouterConfig(nhead=H, kv_heads=KVH, head_dim=Dh, chunk_size=C, group_size=gs,
                       topk_chunks=0, topk_groups=0, theta=10000.0)
    policy = ChunkPlacementPolicy(keep_last=10_000, keep_first=0, first_token_level=True)
    store = VramKVCacheStore(compute_device=torch.device(device), policy=policy, kv_heads=KVH,
                             head_dim=Dh, chunk_size=C, groups_per_chunk=C // gs, batch_size=B,
                             dtype=torch.float32)
    router = ExactChunkRouter(cfg, store)
    router.reset()

    outs = []
    p = 0
    while p < S:
        take = min(C - (p % C), S - p)
        sl = slice(p, p + take)
        routed = router.decode_block_exact(0, q[:, :, sl], k[:, :, sl], k[:, :, sl], v[:, :, sl], p)
        outs.append(routed.attend(q[:, :, sl]))
        p += take
    out = torch.cat(outs, dim=2)

    err = (out - ref).abs().max().item()
    print(f"[selftest] exact-router vs dense causal SDPA  max abs err = {err:.3e}")
    assert err < 1e-4, f"exact router != dense causal ({err})"
    print("[selftest] PASSED")


def selftest_wrapper_vs_qwen(device: str = "cpu", *, realistic: bool = False) -> None:
    """End-to-end: QwenRoutedAttention(exact, all-local) must equal the real Qwen3MoeAttention.

    Exercises the full wrapper (projections, q/k norms, RoPE via apply_rotary_pos_emb, chunk
    streaming, store gather, o_proj) against the stock module — the path the offline
    ``selftest_exact_equivalence`` bypassed.  ``realistic=True`` mirrors the 30B shapes
    (head_dim 128, GQA 32:4, rope_theta 1e7, chunk 64) in bf16 to catch precision/shape bugs.
    """
    import types
    from transformers import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import (
        Qwen3MoeAttention, Qwen3MoeRotaryEmbedding,
    )

    torch.manual_seed(0)
    if realistic:
        cfg = Qwen3MoeConfig(
            hidden_size=2048, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
            num_hidden_layers=1, num_experts=4, num_experts_per_tok=2, rms_norm_eps=1e-6,
            rope_theta=10_000_000.0, attention_bias=False, max_position_embeddings=262144,
        )
        S, C, gs, dtype, tol, tag = 200, 64, 16, torch.bfloat16, 0.05, "realistic-bf16"
    else:
        cfg = Qwen3MoeConfig(
            hidden_size=128, num_attention_heads=8, num_key_value_heads=2, head_dim=16,
            num_hidden_layers=1, num_experts=4, num_experts_per_tok=2, rms_norm_eps=1e-6,
            rope_theta=10000.0, attention_bias=False, max_position_embeddings=4096,
        )
        S, C, gs, dtype, tol, tag = 20, 8, 4, torch.float32, 1e-3, "tiny-fp32"
    cfg._attn_implementation = "eager"
    attn = Qwen3MoeAttention(cfg, layer_idx=0).to(device=device, dtype=dtype).eval()
    rot = Qwen3MoeRotaryEmbedding(cfg).to(device)

    B = 1
    x = torch.randn(B, S, cfg.hidden_size, device=device, dtype=dtype)
    pos = torch.arange(S, device=device).unsqueeze(0)
    cos, sin = rot(x, pos)
    causal = torch.triu(torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1).view(1, 1, S, S)

    with torch.no_grad():
        # Reference = SDPA (fp32 accumulation), which is what the real model uses.
        cfg._attn_implementation = "sdpa"
        ref, _ = attn(x, position_embeddings=(cos, sin), attention_mask=causal, past_key_values=None)
        cfg._attn_implementation = "eager"
        ref_eager, _ = attn(x, position_embeddings=(cos, sin), attention_mask=causal, past_key_values=None)
        print(f"[selftest:{tag}] eager-vs-sdpa gap = {(ref_eager-ref).abs().max().item():.3e} "
              f"(shows why bf16 attend must upcast)")
        w = QwenRoutedAttention(attn, cfg, mode="exact", keep_first=0, keep_last=9999,
                                topk_chunks=0, chunk_size=C, group_size=gs)
        out, _ = w(x, position_embeddings=(cos, sin), attention_mask=None, position_ids=pos)
        # blocked: multiple forward() calls sharing a cache-attached router (the real scenario)
        w2 = QwenRoutedAttention(attn, cfg, mode="exact", keep_first=0, keep_last=9999,
                                 topk_chunks=0, chunk_size=C, group_size=gs)
        pkv = types.SimpleNamespace()
        outs, blk = [], 16
        for s in range(0, S, blk):
            e = min(s + blk, S)
            pids = torch.arange(s, e, device=device).unsqueeze(0)
            ob, _ = w2(x[:, s:e], position_embeddings=(cos[:, s:e], sin[:, s:e]),
                       attention_mask=None, past_key_values=pkv, position_ids=pids)
            outs.append(ob)
        out_blk = torch.cat(outs, dim=1)

    err = (out - ref).abs().max().item()
    err_blk = (out_blk - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"[selftest:{tag}] wrapper single  max abs err = {err:.3e}  (ref scale {scale:.2f})")
    print(f"[selftest:{tag}] wrapper blocked max abs err = {err_blk:.3e}")
    assert err < tol and err_blk < tol, f"wrapper != Qwen attention (single {err}, blocked {err_blk})"
    print(f"[selftest:{tag}] PASSED")


# -------------------------------------------------------------------------------------------------
# Stage 2: three-way comparison on the real model
# -------------------------------------------------------------------------------------------------
SAMPLE = """The history of long-context language modeling is a story of fighting the quadratic
cost of attention. Early transformers attended over every pair of tokens, which is exact but
scales poorly: doubling the sequence quadruples the work and the memory of the key-value cache.
Researchers responded with sparse patterns. Sliding-window attention keeps only a local band,
attention sinks discovered that the first few tokens act as a stabilizing anchor, and block-sparse
methods route each query to a handful of relevant key blocks. The MInference line of work observed
that real attention maps fall into a small number of shapes — an A-shape that keeps initial sink
tokens together with a local window, vertical-and-slash patterns, and dense blocks — and exploits
them to accelerate the prefill stage without retraining. A complementary idea is to push the cold
parts of the key-value cache out of fast memory entirely, keeping only summaries resident and
fetching the exact keys and values of a block on demand when a query is routed to it. This makes
million-token contexts feasible on a single memory-limited accelerator, because the working set is
bounded by the number of routed blocks rather than the full sequence length. The remaining question
is fidelity: does selecting a sparse subset of the context degrade the model's predictions? When the
selection is good and attention is then computed over the real keys and values, the answer is often
no — the dropped blocks contributed little probability mass anyway. """


def build_ids(tok, n_tokens: int, device: str) -> torch.Tensor:
    text = SAMPLE
    ids = tok(text, return_tensors="pt").input_ids
    while ids.shape[1] < n_tokens:
        text = text + "\n\n" + SAMPLE
        ids = tok(text, return_tensors="pt").input_ids
    return ids[:, :n_tokens].to(device)


@torch.inference_mode()
def streamed_predictions(model, ids: torch.Tensor, block: int):
    """Feed the sequence in cache-backed blocks; yield ``(start, logits[1,blk,V])`` per block.

    Blocked + cached forward is mathematically identical to a single full forward for dense
    attention (KV cache = exact causal), and it bounds the MoE/lm_head activation peak so the
    2000-token context fits in the ~3GB free after the 30GB FP8 weights.  For the routed
    attention the router persists on the ``past_key_values`` object across blocks; our attention
    ignores the (empty) standard cache, so no duplicate KV is stored.
    """
    from transformers import DynamicCache
    S = ids.shape[1]
    device = ids.device
    cache = DynamicCache()
    for s in range(0, S, block):
        e = min(s + block, S)
        cp = torch.arange(s, e, device=device)
        out = model(input_ids=ids[:, s:e], past_key_values=cache, cache_position=cp,
                    position_ids=cp.unsqueeze(0), use_cache=True)
        yield s, out.logits


class Metric:
    """Incremental greedy-match (vs. a reference) + perplexity, never holding full logits."""

    def __init__(self, S: int, tail_start: int, ref_pred: torch.Tensor | None) -> None:
        self.S = S
        self.tail_start = tail_start
        self.ref_pred = ref_pred                      # CPU long [S-1] or None (this *is* the ref)
        self.pred = torch.empty(S - 1, dtype=torch.long)  # collected predictions (CPU)
        self.ce_sum = 0.0
        self.n = 0
        self.agree = 0
        self.agree_tail = 0
        self.n_tail = 0
        self.bucket_edges = [0, 64, 128, 192, 256, 320, 512, 1024, 2048]
        self.bucket_agree = [0] * len(self.bucket_edges)
        self.bucket_n = [0] * len(self.bucket_edges)

    def add(self, s: int, logits: torch.Tensor, ids: torch.Tensor) -> None:
        lg = logits[0].float()                        # [blk, V]
        blk = lg.shape[0]
        lim = min(blk, self.S - 1 - s)                # positions p in [s, s+lim) predict p+1
        if lim <= 0:
            return
        lg = lg[:lim]
        pred = lg.argmax(-1)                          # [lim] on device
        tgt = ids[0, s + 1: s + 1 + lim]
        self.ce_sum += F.cross_entropy(lg, tgt, reduction="sum").item()
        self.n += lim
        pred_cpu = pred.to("cpu")
        self.pred[s: s + lim] = pred_cpu
        if self.ref_pred is not None:
            ref = self.ref_pred[s: s + lim]
            eq = (pred_cpu == ref)
            self.agree += int(eq.sum())
            tmask = torch.arange(s, s + lim) >= self.tail_start
            if tmask.any():
                self.agree_tail += int((eq & tmask).sum())
                self.n_tail += int(tmask.sum())
            for p in range(s, s + lim):
                bi = max(i for i, e in enumerate(self.bucket_edges) if p >= e)
                self.bucket_n[bi] += 1
                self.bucket_agree[bi] += int(eq[p - s])

    @property
    def ppl(self) -> float:
        return math.exp(self.ce_sum / max(1, self.n))

    def line(self, name: str) -> str:
        if self.ref_pred is None:
            return (f"  {name:8s}  greedy-match(all)=100.00%  greedy-match(tail)=100.00%  "
                    f"ppl={self.ppl:7.3f}")
        a = 100.0 * self.agree / max(1, self.n)
        at = 100.0 * self.agree_tail / max(1, self.n_tail)
        buckets = "  ".join(
            f"{self.bucket_edges[i]}:{100.0*self.bucket_agree[i]/n:.0f}%"
            for i, n in enumerate(self.bucket_n) if n)
        return (f"  {name:14s}  match(all)={a:6.2f}%  ppl={self.ppl:8.3f}\n"
                f"                  by-pos[{buckets}]")


def compare_on_qwen(args) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.reset_peak_memory_stats()
    device = "cuda"

    print(f"[load] {MODEL}", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype="auto", device_map="cuda", attn_implementation="sdpa",
    ).eval()
    torch.cuda.synchronize()
    print(f"[load] done in {time.perf_counter()-t0:.1f}s  "
          f"({gb(torch.cuda.memory_allocated()):.1f}GB allocated)", flush=True)

    ids = build_ids(tok, args.tokens, device)
    S = ids.shape[1]
    tail_start = (args.keep_first + args.keep_last) * 64  # routing becomes active past the windows
    tail_start = min(tail_start, S - 2)
    print(f"[data] context = {S} tokens, block = {args.block}; routing active for query positions > {tail_start}\n"
          f"[cfg ] keep_first={args.keep_first} ({args.keep_first*64} sink tok), "
          f"keep_last={args.keep_last} ({args.keep_last*64} local tok), topk_chunks={args.topk} "
          f"=> {(args.keep_first+args.keep_last+args.topk)} active chunks / {S//64} total\n",
          flush=True)

    def run(name: str, ref_pred) -> Metric:
        m = Metric(S, tail_start, ref_pred)
        t = time.perf_counter()
        for s, logits in streamed_predictions(model, ids, args.block):
            m.add(s, logits, ids)
        torch.cuda.synchronize()
        print(f"[run ] {name} {time.perf_counter()-t:.1f}s", flush=True)
        torch.cuda.empty_cache()
        return m

    # --- baseline (reference) ---
    base = run("baseline", None)
    base_pred = base.pred  # CPU long [S-1]

    # --- summary router (the existing HierarchicalGlobalAttentionRouted approach) ---
    n = replace_qwen_attention_with_router(
        model, mode="summary", keep_first=args.keep_first, keep_last=args.keep_last,
        topk_chunks=args.topk)
    summ = run(f"summary ({n} layers)", base_pred)

    # --- exact router (the new approach) ---
    n = replace_qwen_attention_with_router(
        model, mode="exact", keep_first=args.keep_first, keep_last=args.keep_last,
        topk_chunks=args.topk)
    exact = run(f"exact ({n} layers)", base_pred)

    restore_original_attention(model)

    print("\nResults (greedy-match measured against baseline):")
    print(base.line("baseline"))
    print(summ.line("summary"))
    print(exact.line("exact"))
    print(f"\n[mem ] peak allocated = {gb(torch.cuda.max_memory_allocated()):.1f}GB / "
          f"{gb(torch.cuda.get_device_properties(0).total_memory):.1f}GB")


RELEVANT = ("Important facts to remember. The launch code is ZEBRA-7. "
            "The project review meeting is scheduled for Friday at 10 AM in room 250. "
            "The lead engineer on the project is Dr. Maria Chen.")
QUESTION = ("\n\nUsing only the important facts above, answer concisely.\n"
            "Question: What is the launch code, on what day and time is the review meeting, "
            "and who is the lead engineer?\nAnswer:")


def filler_text(tok, n_tokens: int) -> str:
    """Unrelated filler of about ``n_tokens`` tokens (tiled SAMPLE essay)."""
    text = SAMPLE
    while len(tok(text).input_ids) < n_tokens:
        text = text + "\n\n" + SAMPLE
    ids = tok(text).input_ids[:n_tokens]
    return tok.decode(ids)


@torch.inference_mode()
def greedy_generate(model, tok, prompt_ids: torch.Tensor, max_new: int, block: int):
    """Blocked, cache-backed greedy decode. Works for both the dense baseline (real attention
    updates the cache) and the routed model (router persists on the cache; cache stays empty)."""
    from transformers import DynamicCache
    device = prompt_ids.device
    eos = tok.eos_token_id if isinstance(tok.eos_token_id, int) else None
    cache = DynamicCache()
    S = prompt_ids.shape[1]
    last = None
    for s in range(0, S, block):
        e = min(s + block, S)
        cp = torch.arange(s, e, device=device)
        out = model(input_ids=prompt_ids[:, s:e], past_key_values=cache, cache_position=cp,
                    position_ids=cp.unsqueeze(0), use_cache=True)
        last = out.logits[:, -1]
    gen, p = [], S
    nxt = int(last.argmax(-1))
    for _ in range(max_new):
        if nxt == eos:
            break
        gen.append(nxt)
        cp = torch.tensor([p], device=device)
        out = model(input_ids=torch.tensor([[nxt]], device=device), past_key_values=cache,
                    cache_position=cp, position_ids=cp.unsqueeze(0), use_cache=True)
        nxt = int(out.logits[:, -1].argmax(-1))
        p += 1
    return tok.decode(gen).strip()


def compare_ram(args) -> None:
    """RAM-cache test: a fact + question sit at the end; a long *irrelevant* prefix precedes them
    and lives in host RAM (only routed chunks are pulled to VRAM).  The routed answer should match
    the dense baseline's answer to the same fact+question, and VRAM stays bounded at 32K context."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available()
    device = "cuda"
    print(f"[load] {MODEL}", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype="auto", device_map="cuda", attn_implementation="sdpa").eval()
    torch.cuda.synchronize()
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)

    rk = dict(keep_first=args.keep_first, keep_last=args.keep_last, topk_chunks=args.topk,
              cache_location="ram")
    print(f"[cfg ] mode=exact RAM cache  keep_first={args.keep_first} keep_last={args.keep_last} "
          f"topk_chunks={args.topk}  block={args.block}\n", flush=True)

    # --- dense baseline answer (no irrelevant prefix) ---
    restore_original_attention(model)
    base_ids = tok(RELEVANT + QUESTION, return_tensors="pt").input_ids.to(device)
    base_ans = greedy_generate(model, tok, base_ids, args.max_new, args.block)
    print(f"[baseline dense]  ({base_ids.shape[1]} tok)\n  -> {base_ans!r}\n", flush=True)

    # --- routed RAM-cache answers with growing irrelevant prefix ---
    for ctx in args.ctx_sizes:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        prefix = filler_text(tok, ctx)
        ids = tok(prefix + "\n\n" + RELEVANT + QUESTION, return_tensors="pt").input_ids.to(device)
        n = replace_qwen_attention_with_router(model, mode="exact", **rk)
        t = time.perf_counter()
        ans = greedy_generate(model, tok, ids, args.max_new, args.block)
        dt = time.perf_counter() - t
        peak = gb(torch.cuda.max_memory_allocated())
        restore_original_attention(model)
        print(f"[routed RAM ~{ids.shape[1]} tok]  ({n} layers, {dt:.0f}s, peak {peak:.1f}GB)\n"
              f"  -> {ans!r}", flush=True)
        for needle in ("ZEBRA-7", "Friday", "10", "250", "Chen"):
            mark = "ok" if needle.lower() in ans.lower() else "MISS"
            print(f"     [{mark}] {needle}", flush=True)
        print(flush=True)


def diag_sweep(args) -> None:
    """Load once; run baseline then exact-router at several window configs to localize errors."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available()
    torch.cuda.reset_peak_memory_stats()
    device = "cuda"
    print(f"[load] {MODEL}", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype="auto", device_map="cuda", attn_implementation="sdpa").eval()
    torch.cuda.synchronize()
    print(f"[load] done in {time.perf_counter()-t0:.1f}s", flush=True)

    ids = build_ids(tok, args.tokens, device)
    S = ids.shape[1]

    def run(name, ref):
        m = Metric(S, 256, ref)
        for s, logits in streamed_predictions(model, ids, args.block):
            m.add(s, logits, ids)
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        print(m.line(name), flush=True)
        return m

    base = run("baseline", None)
    bp = base.pred
    configs = [
        ("kf0_kl999_t0", dict(keep_first=0, keep_last=999, topk_chunks=0)),
        ("kf2_kl999_t0", dict(keep_first=2, keep_last=999, topk_chunks=0)),
        ("kf0_kl2_t999", dict(keep_first=0, keep_last=2, topk_chunks=999)),
        ("kf2_kl2_t8",   dict(keep_first=2, keep_last=2, topk_chunks=8)),
    ]
    for label, cfg in configs:
        replace_qwen_attention_with_router(model, mode="exact", **cfg)
        run(f"exact:{label}", bp)
    restore_original_attention(model)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--block", type=int, default=128, help="prefill block size (multiple of 64)")
    ap.add_argument("--keep-first", type=int, default=2)
    ap.add_argument("--keep-last", type=int, default=2)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--selftest-only", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="localize errors across window configs")
    ap.add_argument("--ram", action="store_true", help="RAM-cache irrelevant-prefix / 32K test")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--ctx-sizes", type=int, nargs="+", default=[2048, 32768],
                    help="irrelevant-prefix context sizes for the RAM test")
    args = ap.parse_args()

    selftest_exact_equivalence("cpu")
    selftest_wrapper_vs_qwen("cpu", realistic=False)
    selftest_wrapper_vs_qwen("cpu", realistic=True)
    if args.selftest_only:
        return
    print()
    if args.ram:
        compare_ram(args)
    elif args.sweep:
        diag_sweep(args)
    else:
        compare_on_qwen(args)


if __name__ == "__main__":
    main()
