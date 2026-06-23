#!/usr/bin/env python3
"""Tests for DCA: patched decoder position_embeddings + three-path attend in routed attention.

Run:
    python -m ExistingModelFineTuning.Qwen3LongContext.Tests.test_dca_long_context
    python -m ExistingModelFineTuning.Qwen3LongContext.Tests.test_dca_long_context --cuda
"""

from __future__ import annotations

import argparse

import torch
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeRotaryEmbedding,
    apply_rotary_pos_emb,
)

from ExistingModelFineTuning.Qwen3LongContext.dca_rope import (
    build_dca_embeddings,
    infer_long_context_mode,
    patch_qwen_rotary_emb,
    resolve_long_context_settings,
    restore_qwen_rotary_emb,
)
from ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention import (
    QwenRoutedAttention,
    replace_qwen_attention_with_router,
    restore_original_attention,
)


def _tiny_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        attention_bias=False,
        max_position_embeddings=262144,
    )


class _FakeModel(torch.nn.Module):
    def __init__(self, cfg: Qwen3MoeConfig):
        super().__init__()
        self.config = cfg
        self.model = torch.nn.Module()
        self.model.rotary_emb = Qwen3MoeRotaryEmbedding(cfg)
        self.model.layers = torch.nn.ModuleList()


def test_mode_selection() -> None:
    cfg = _tiny_config()
    assert infer_long_context_mode(cfg, target_context=262144) == "native"
    assert infer_long_context_mode(cfg, target_context=1_000_000) == "hybrid"
    assert infer_long_context_mode(cfg, target_context=1_000_000, force_dca=True) == "dca"


def test_rotary_patch_dca_embeddings(device: str) -> None:
    cfg = _tiny_config()
    model = _FakeModel(cfg).to(device)
    x = torch.randn(1, 4, cfg.hidden_size, device=device)
    abs_pos = torch.tensor([[100, 262143, 262144, 500_000]], device=device)

    lc = resolve_long_context_settings(cfg, target_context=1_000_000)
    cl = lc.dca.chunk_len

    patch_qwen_rotary_emb(model, lc)
    dca_cos, dca_sin = model.model.rotary_emb(x, position_ids=abs_pos)

    restore_qwen_rotary_emb(model)
    for i, p in enumerate(abs_pos[0].tolist()):
        ref_pos = torch.tensor([[p % cl]], device=device)
        ref_cos, ref_sin = model.model.rotary_emb(x, position_ids=ref_pos)
        assert torch.allclose(dca_cos[:, i : i + 1], ref_cos, atol=1e-5), f"pos {p}"
        assert torch.allclose(dca_sin[:, i : i + 1], ref_sin, atol=1e-5), f"pos {p}"

    native_cos, _ = model.model.rotary_emb(x, position_ids=abs_pos)
    assert not torch.allclose(dca_cos[:, 3:4], native_cos[:, 3:4], atol=1e-3)
    print("[patch] DCA position_embeddings match cyclic rotary_emb")


def test_dca_tables_match_patched_decoder(device: str) -> None:
    cfg = _tiny_config()
    lc = resolve_long_context_settings(cfg, target_context=1_000_000)
    rot = Qwen3MoeRotaryEmbedding(cfg).to(device)
    x = torch.randn(1, 1, cfg.hidden_size, device=device)
    emb = build_dca_embeddings(rot, x, lc.dca)
    pos = torch.tensor([[42]], device=device)
    dec_cos, _ = rot(x, pos.remainder(lc.dca.chunk_len))
    assert torch.allclose(emb.q_cos[42:43], dec_cos, atol=1e-5)
    assert torch.allclose(emb.k_cos[42:43], dec_cos, atol=1e-5)
    print("[tables] qc tables align with decoder rotary_emb")


def test_routed_native_equivalence(device: str) -> None:
    cfg = _tiny_config()
    cfg._attn_implementation = "eager"
    S, C, gs = 20, 8, 4
    dtype = torch.float32
    tol = 1e-3

    attn = Qwen3MoeAttention(cfg, layer_idx=0).to(device=device, dtype=dtype).eval()
    rot = Qwen3MoeRotaryEmbedding(cfg).to(device)
    x = torch.randn(1, S, cfg.hidden_size, device=device, dtype=dtype)
    pos = torch.arange(S, device=device).unsqueeze(0)
    cos, sin = rot(x, pos)
    lc = resolve_long_context_settings(cfg, target_context=262144)

    causal = torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1,
    ).view(1, 1, S, S)

    with torch.no_grad():
        ref, _ = attn(x, position_embeddings=(cos, sin), attention_mask=causal, past_key_values=None)
        w = QwenRoutedAttention(
            attn, cfg, keep_first=0, keep_last=9999, topk_chunks=0, topk_groups=0,
            chunk_size=C, group_size=gs, long_context=lc, rotary_emb=rot,
        )
        out, _ = w(x, position_embeddings=(cos, sin), attention_mask=None, position_ids=pos)
        err = (out - ref).abs().max().item()
        assert err < tol, f"routed native != ref ({err})"
    print(f"[routed-native] matches Qwen attention (max err {err:.3e})")


def test_routed_dca_first_chunk(device: str) -> None:
    cfg = _tiny_config()
    cfg._attn_implementation = "eager"
    S, C, gs = 16, 8, 4
    dtype = torch.float32
    tol = 1e-3

    attn = Qwen3MoeAttention(cfg, layer_idx=0).to(device=device, dtype=dtype).eval()
    rot = Qwen3MoeRotaryEmbedding(cfg).to(device)
    fake = _FakeModel(cfg).to(device)
    fake.model.rotary_emb = rot

    x = torch.randn(1, S, cfg.hidden_size, device=device, dtype=dtype)
    pos = torch.arange(S, device=device).unsqueeze(0)
    native_cos, native_sin = rot(x, pos)

    lc = resolve_long_context_settings(cfg, target_context=1_000_000)
    patch_qwen_rotary_emb(fake, lc)
    dca_cos, dca_sin = rot(x, pos)

    causal = torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=dtype), diagonal=1,
    ).view(1, 1, S, S)

    with torch.no_grad():
        ref, _ = attn(
            x, position_embeddings=(native_cos, native_sin),
            attention_mask=causal, past_key_values=None,
        )
        w = QwenRoutedAttention(
            attn, cfg, keep_first=0, keep_last=9999, topk_chunks=0, topk_groups=0,
            chunk_size=C, group_size=gs, long_context=lc, rotary_emb=rot,
        )
        assert w.use_dca
        out, _ = w(x, position_embeddings=(dca_cos, dca_sin), attention_mask=None, position_ids=pos)
        err = (out - ref).abs().max().item()
        assert err < tol, f"DCA first-chunk != native ({err})"
    restore_qwen_rotary_emb(fake)
    print(f"[routed-dca] first-chunk matches native (max err {err:.3e})")


class _DecoderLayer(torch.nn.Module):
    def __init__(self, attn: Qwen3MoeAttention) -> None:
        super().__init__()
        self.self_attn = attn


def test_replace_and_restore(device: str) -> None:
    cfg = _tiny_config()
    cfg.num_hidden_layers = 2
    attn0 = Qwen3MoeAttention(cfg, layer_idx=0)
    attn1 = Qwen3MoeAttention(cfg, layer_idx=1)
    layer0 = _DecoderLayer(attn0)
    layer1 = _DecoderLayer(attn1)
    model = _FakeModel(cfg)
    model.model.layers = torch.nn.ModuleList([layer0, layer1])

    n = replace_qwen_attention_with_router(model, target_context=1_000_000)
    assert n == 2
    assert hasattr(model.model.rotary_emb, "_dca_orig_forward")
    assert isinstance(layer0.self_attn, QwenRoutedAttention)
    assert layer0.self_attn.use_dca

    m = restore_original_attention(model)
    assert m == 2
    assert not hasattr(model.model.rotary_emb, "_dca_orig_forward")
    assert layer0.self_attn is attn0
    print("[replace] patch + restore round-trip OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"device={device}\n")

    test_mode_selection()
    test_rotary_patch_dca_embeddings(device)
    test_dca_tables_match_patched_decoder(device)
    test_routed_native_equivalence(device)
    test_routed_dca_first_chunk(device)
    test_replace_and_restore(device)
    print("\nAll DCA tests PASSED")


if __name__ == "__main__":
    main()