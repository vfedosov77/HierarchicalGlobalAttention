"""Offline selftests for the Ornith (Qwen3.5-hybrid) routed attention surgery.

Run:  python -m ExistingModelFineTuning.OrnithLongContext.Tests.test_ornith_routed

No model download: everything runs on a tiny synthetic Qwen3.5 text config.

Covered:
* A  — wrapper == stock ``Qwen3_5Attention`` (gate + partial-RoPE + GQA) at full routing
       coverage, to <1e-4.
* B1 — full 4-layer all-full text model: wrapped (full coverage) == stock logits + reset.
* B2 — hybrid stack (linear+full): navigation wraps *only* full layers (idx 3, 7), linear
       untouched, first_attn_layer_idx == 3, restore works.
* C  — partial RoPE (rotary_dim < head_dim): router runs and rotary tables have width
       rotary_dim.
"""

import torch

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
)

from ..ornith_routed_attention import (
    OrnithRoutedAttention,
    _iter_ornith_attention_layers,
    replace_ornith_attention_with_router,
    restore_ornith_attention,
)
from ...KvRouter.cache_store import ChunkPlacementPolicy, RamKVCacheStore
from ...KvRouter.chunk_router import ChunkRouter, RouterConfig

_PARTIAL = 0.25


def _tiny_cfg(layer_types, *, head_dim=64, hidden=128, heads=4, kv=2, vocab=256):
    cfg = Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=2 * hidden,
        num_hidden_layers=len(layer_types), num_attention_heads=heads,
        num_key_value_heads=kv, head_dim=head_dim, rms_norm_eps=1e-6,
        attention_bias=False, attention_dropout=0.0, layer_types=list(layer_types),
        tie_word_embeddings=False,
        rope_parameters={
            "rope_type": "default", "rope_theta": 10000.0,
            "partial_rotary_factor": _PARTIAL, "mrope_section": [3, 3, 2],
        },
        linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4, use_cache=True,
    )
    cfg._attn_implementation = "eager"
    return cfg


def _force_full_coverage(module: OrnithRoutedAttention) -> None:
    """Make the router reproduce plain dense causal attention (for equivalence checks)."""
    module._cfg.current_group_summaries = False
    module._cfg.topk_chunks = 0
    module._cfg.topk_groups = 0
    module._policy = ChunkPlacementPolicy(keep_last=0, keep_first=999, first_token_level=True)


# ---------------------------------------------------------------------------
def test_wrapper_matches_stock_attention(device):
    """A: OrnithRoutedAttention (full coverage) == stock Qwen3_5Attention to <1e-4."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(["full_attention"])
    B, S = 2, 40
    Dh = cfg.head_dim
    dtype = torch.float32

    orig = Qwen3_5Attention(cfg, 0).to(device=device, dtype=dtype).eval()
    rot = Qwen3_5TextRotaryEmbedding(cfg).to(device)
    hidden = torch.randn(B, S, cfg.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(S, device=device).unsqueeze(0)          # [1, S]
    cos, sin = rot(hidden, position_ids)

    causal = torch.triu(torch.full((S, S), float("-inf"), device=device, dtype=dtype), 1).view(1, 1, S, S)
    with torch.no_grad():
        ref, _ = orig(hidden_states=hidden, position_embeddings=(cos, sin),
                      attention_mask=causal, past_key_values=None)

    w = OrnithRoutedAttention(
        orig, cfg, first_attn_layer_idx=0, num_wrapped_layers=1,
        chunk_size=16, group_size=8, cache_location="ram",
    ).to(device)
    _force_full_coverage(w)
    with torch.no_grad():
        out, _ = w(hidden_states=hidden, position_embeddings=(cos, sin),
                   attention_mask=None, past_key_values=None, position_ids=position_ids)

    err = (out - ref).abs().max().item()
    print(f"[wrapper_vs_stock] max abs err = {err:.3e}  rotary_dim={w._cfg.rotary_dim_resolved}/{Dh}")
    assert w._cfg.rotary_dim_resolved == round(Dh * _PARTIAL), w._cfg.rotary_dim_resolved
    assert err < 1e-4, err


def test_full_model_all_full(device):
    """B1: 4-layer all-full text model, wrapped (full coverage) == stock last_hidden_state."""
    torch.manual_seed(1)
    cfg = _tiny_cfg(["full_attention"] * 4)
    model = Qwen3_5TextModel(cfg).to(device).eval()
    B, S = 2, 40
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=device)

    with torch.no_grad():
        ref = model(input_ids=input_ids, use_cache=False).last_hidden_state

    n = replace_ornith_attention_with_router(model, chunk_size=16, group_size=8, cache_location="ram")
    for layer in _iter_ornith_attention_layers(model):
        _force_full_coverage(layer.self_attn)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False).last_hidden_state

    err = (out - ref).abs().max().item()
    firsts = {layer.self_attn.first_attn_layer_idx for layer in _iter_ornith_attention_layers(model)}
    print(f"[full_model_all_full] wrapped {n} layers, first={firsts}, max abs err = {err:.3e}")
    assert n == 4, n
    assert firsts == {0}, firsts
    assert err < 1e-4, err
    assert restore_ornith_attention(model) == 4


def test_hybrid_structure(device):
    """B2: linear+full stack — only full layers (idx 3, 7) wrapped, linear untouched."""
    torch.manual_seed(2)
    layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 2   # 8 layers, full at 3, 7
    cfg = _tiny_cfg(layer_types)
    model = Qwen3_5TextModel(cfg).to(device).eval()

    full_before = [i for i, l in enumerate(model.layers) if hasattr(l, "self_attn")]
    linear_idx = [i for i, l in enumerate(model.layers) if hasattr(l, "linear_attn")]

    n = replace_ornith_attention_with_router(model, chunk_size=16, group_size=8, cache_location="ram")
    firsts = {model.layers[i].self_attn.first_attn_layer_idx for i in full_before}
    print(f"[hybrid] full idx={full_before}, linear idx={linear_idx}, wrapped={n}, first={firsts}")

    assert full_before == [3, 7], full_before
    assert n == 2, n
    assert firsts == {3}, firsts
    assert all(isinstance(model.layers[i].self_attn, OrnithRoutedAttention) for i in full_before)
    # linear layers must be entirely untouched
    assert all(hasattr(model.layers[i], "linear_attn") for i in linear_idx)
    assert all(not hasattr(model.layers[i], "self_attn") for i in linear_idx)

    assert restore_ornith_attention(model) == 2
    assert all(not isinstance(model.layers[i].self_attn, OrnithRoutedAttention) for i in full_before)


def test_partial_rotary_router_shapes(device):
    """C: rotary_dim < head_dim — router runs and rotary tables have width rotary_dim."""
    cfg = RouterConfig(
        nhead=4, kv_heads=2, head_dim=64, chunk_size=16, group_size=8,
        topk_chunks=2, topk_groups=2, theta=10000.0, rotary_dim=16,
    )
    assert cfg.rotary_dim_resolved == 16
    B, S = 1, 48
    store = RamKVCacheStore(
        compute_device=torch.device(device),
        policy=ChunkPlacementPolicy(keep_last=1, keep_first=1, first_token_level=False),
        kv_heads=cfg.kv_heads, head_dim=cfg.head_dim, chunk_size=cfg.chunk_size,
        groups_per_chunk=cfg.groups_per_chunk, batch_size=B, dtype=torch.float32, pin_memory=False,
    )
    router = ChunkRouter(cfg, store)
    g = torch.Generator(device=device).manual_seed(0)
    q = torch.randn(B, cfg.nhead, S, cfg.head_dim, generator=g, device=device)
    k = torch.randn(B, cfg.kv_heads, S, cfg.head_dim, generator=g, device=device)
    v = torch.randn(B, cfg.kv_heads, S, cfg.head_dim, generator=g, device=device)
    out = router.prefill(0, q, k, k, v, start_pos=0)
    cos, sin = router.rotary_table(0, S, torch.device(device))
    print(f"[partial_rotary] out shape={tuple(out.shape)} finite={torch.isfinite(out).all().item()} "
          f"rotary_table width={cos.shape[-1]}")
    assert out.shape == (B, cfg.nhead, S, cfg.head_dim)
    assert torch.isfinite(out).all()
    assert cos.shape[-1] == 16 and sin.shape[-1] == 16


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")
    test_wrapper_matches_stock_attention(dev)
    test_full_model_all_full(dev)
    test_hybrid_structure(dev)
    test_partial_rotary_router_shapes(dev)
    print("ALL PASSED")
