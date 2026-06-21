import torch, math
from HierarchicalGlobalAttentionFusedExactQ_cleaned import GlobalAttention

torch.manual_seed(0)
dev = "cuda"


class FakeCache:
    """Minimal Qwen-Cache-like object: per-layer key/value concatenation."""
    def __init__(self):
        self.k = {}
        self.v = {}
    def update(self, k, v, layer_idx, cache_kwargs=None):
        if layer_idx in self.k:
            self.k[layer_idx] = torch.cat([self.k[layer_idx], k], dim=2)
            self.v[layer_idx] = torch.cat([self.v[layer_idx], v], dim=2)
        else:
            self.k[layer_idx] = k
            self.v[layer_idx] = v
        return self.k[layer_idx], self.v[layer_idx]
    def get_seq_length(self, layer_idx=0):
        return 0 if layer_idx not in self.k else self.k[layer_idx].shape[2]


def make_attn(dtype, **kw):
    cfg = dict(d_model=256, nhead=4, kv_heads=2, head_dim=64, qk_norm=True,
               chunk_size=64, group_size=16, topk_chunks=4, topk_groups=8, layer_idx=0)
    cfg.update(kw)
    a = GlobalAttention(**cfg).to(dev, dtype)
    a.eval()
    return a


def pos_emb(a, S, offset=0, B=1, dtype=torch.float32):
    cos, sin = a._get_rotary(S + offset, dev, torch.float32)
    cos = cos[offset:offset + S].unsqueeze(0).expand(B, S, -1)
    sin = sin[offset:offset + S].unsqueeze(0).expand(B, S, -1)
    return cos, sin


def test_fused_vs_dense():
    # fused needs M>=16 pow2 -> chunk 256 group 16, head_dim 64
    a = make_attn(torch.float32, d_model=256, nhead=4, kv_heads=2, head_dim=64,
                  chunk_size=256, group_size=16, topk_chunks=3, topk_groups=8)
    a.fp32_precise = True
    B, S = 1, 600
    x = torch.randn(B, S, 256, device=dev)
    pe = pos_emb(a, S)
    assert a._fused_applicable(), "fused should apply"
    with torch.no_grad():
        of = a._forward_no_cache(*a._prep(x, pe))[0] if hasattr(a, "_prep") else None
    # call full forward twice forcing fused / dense
    with torch.no_grad():
        q_raw, k_raw, v = a._project_qkv_base(x)
        cos, sin = a._normalize_position_embeddings(pe, x)
        q = a._apply_rotary(q_raw.float(), cos, sin).to(q_raw.dtype)
        k = a._apply_rotary(k_raw.float(), cos, sin).to(k_raw.dtype)
        out_fused = a._forward_fused(q, a._repeat_kv(k), a._repeat_kv(k_raw), a._repeat_kv(v), cos, sin)
        out_dense = a._forward_dense(q_raw, q, a._repeat_kv(k_raw), a._repeat_kv(k), a._repeat_kv(v), cos, sin, None)[0]
    err = (out_fused - out_dense).abs().max().item()
    rel = err / out_dense.abs().max().item()
    print(f"[fused vs dense] max_abs={err:.3e} rel={rel:.3e}  path={a._last_path}")
    assert rel < 1e-2, "fused and dense disagree"


def test_decode_consistency():
    a = make_attn(torch.float32)
    B, S = 1, 220
    x = torch.randn(B, S, 256, device=dev)

    # Reference: dense teacher forcing over full sequence.
    pe = pos_emb(a, S)
    with torch.no_grad():
        out_tf, _ = a.forward(x, pe, past_key_value=None)

    # Token-by-token generation (prefill of 1 then decode), empty cache.
    cache = FakeCache()
    outs = []
    with torch.no_grad():
        for t in range(S):
            pe_t = pos_emb(a, 1, offset=t)
            cp = torch.arange(t, t + 1, device=dev)
            o, _ = a.forward(x[:, t:t + 1], pe_t, past_key_value=cache, cache_position=cp)
            outs.append(o)
    out_gen = torch.cat(outs, dim=1)
    gap = (out_tf - out_gen).abs()
    print(f"[tf vs token-by-token] mean_abs={gap.mean():.4e} max_abs={gap.max():.4e} "
          f"scale={out_tf.abs().mean():.4e}")

    # Prefill of P>1 then decode the rest must match pure token-by-token.
    P = 150
    cache2 = FakeCache()
    with torch.no_grad():
        o_pref, _ = a.forward(x[:, :P], pos_emb(a, P), past_key_value=cache2,
                              cache_position=torch.arange(P, device=dev))
        outs2 = [o_pref]
        for t in range(P, S):
            o, _ = a.forward(x[:, t:t + 1], pos_emb(a, 1, offset=t), past_key_value=cache2,
                             cache_position=torch.arange(t, t + 1, device=dev))
            outs2.append(o)
    out_split = torch.cat(outs2, dim=1)
    d = (out_split[:, P:] - out_gen[:, P:]).abs()
    print(f"[prefill+decode vs token-by-token, decoded part] mean_abs={d.mean():.4e} max_abs={d.max():.4e}")
    assert d.max() < 1e-3, "prefill+decode inconsistent with token-by-token"


def test_overflow_all():
    a = make_attn(torch.float32, decode_route_overflow="all")
    B, S = 1, 200
    x = torch.randn(B, S, 256, device=dev)
    cache = FakeCache()
    with torch.no_grad():
        for t in range(S):
            o, _ = a.forward(x[:, t:t + 1], pos_emb(a, 1, offset=t), past_key_value=cache,
                             cache_position=torch.arange(t, t + 1, device=dev))
    assert torch.isfinite(o).all()
    print("[overflow=all] finite, ok")


if __name__ == "__main__":
    test_fused_vs_dense()
    test_decode_consistency()
    test_overflow_all()
    print("ALL OK")
