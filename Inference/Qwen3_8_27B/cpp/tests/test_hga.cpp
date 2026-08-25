#include "hga.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static float urand(uint32_t & s) {
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xffffff) / 16777216.f * 2.f - 1.f;
}

static void fill(std::vector<float> & a, uint32_t seed) {
    for (float & x : a) x = urand(seed);
}

static float max_abs(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

/* Naive causal SDPA, GQA. q [H,S,D], k/v [KVH,S,D] */
static void dense_sdpa(const float * q, const float * k, const float * v,
                       int H, int KVH, int S, int D, float * out) {
    const int rep = H / KVH;
    const float scale = 1.f / std::sqrt((float)D);
    std::vector<float> scores((size_t)S);
    for (int h = 0; h < H; ++h) {
        const int kh = h / rep;
        for (int t = 0; t < S; ++t) {
            const float * qt = q + ((size_t)h * (size_t)S + (size_t)t) * (size_t)D;
            float m = -1e30f;
            for (int j = 0; j <= t; ++j) {
                const float * kj = k + ((size_t)kh * (size_t)S + (size_t)j) * (size_t)D;
                float s = 0.f;
                for (int d = 0; d < D; ++d) s += qt[d] * kj[d];
                scores[(size_t)j] = s * scale;
                m = std::max(m, scores[(size_t)j]);
            }
            float z = 0.f;
            for (int j = 0; j <= t; ++j) {
                scores[(size_t)j] = std::exp(scores[(size_t)j] - m);
                z += scores[(size_t)j];
            }
            float * ot = out + ((size_t)h * (size_t)S + (size_t)t) * (size_t)D;
            for (int d = 0; d < D; ++d) ot[d] = 0.f;
            for (int j = 0; j <= t; ++j) {
                const float p = scores[(size_t)j] / z;
                const float * vj = v + ((size_t)kh * (size_t)S + (size_t)j) * (size_t)D;
                for (int d = 0; d < D; ++d) ot[d] += p * vj[d];
            }
        }
    }
}

static int fail(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

int main() {
    int nfail = 0;

    /* ---- dense equivalence: keep_first covers everything ---- */
    {
        hga_config cfg = hga_config_qwen38_27b(1, 256, 2);
        cfg.n_q_heads = 8;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 32;
        cfg.rotary_dim = 16;
        cfg.chunk_size = 8;
        cfg.group_size = 4;
        cfg.keep_first = 999;
        cfg.keep_last = 0;
        cfg.frac_l1 = 1.f;
        cfg.theta = 10000.f;
        cfg.prec = HGA_PREC_F16;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim, S = 24;
        hga_session * sess = hga_session_create(&cfg, 1);
        std::vector<float> q((size_t)H * S * D), k((size_t)KVH * S * D), v((size_t)KVH * S * D),
            o_hga((size_t)H * S * D), o_ref((size_t)H * S * D);
        fill(q, 1); fill(k, 2); fill(v, 3);
        /* Stream one token at a time so the active-chunk causal path is exercised. */
        for (int t = 0; t < S; ++t) {
            std::vector<float> qt((size_t)H * D), kt((size_t)KVH * D), vt((size_t)KVH * D);
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d)
                    qt[(size_t)h * D + d] = q[((size_t)h * S + t) * D + d];
            for (int h = 0; h < KVH; ++h)
                for (int d = 0; d < D; ++d) {
                    kt[(size_t)h * D + d] = k[((size_t)h * S + t) * D + d];
                    vt[(size_t)h * D + d] = v[((size_t)h * S + t) * D + d];
                }
            std::vector<float> ot((size_t)H * D);
            hga_forward(sess, 0, t, 1, qt.data(), kt.data(), kt.data(), vt.data(), HGA_F32, ot.data(), nullptr);
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d)
                    o_hga[((size_t)h * S + t) * D + d] = ot[(size_t)h * D + d];
        }
        dense_sdpa(q.data(), k.data(), v.data(), H, KVH, S, D, o_ref.data());
        const float err = max_abs(o_hga, o_ref);
        std::printf("[dense_equivalence] max abs err = %.3e\n", err);
        if (!(err < 2e-3f)) { std::fprintf(stderr, "  too large\n"); ++nfail; }
        hga_session_free(sess);
    }

    /* ---- same chunk count, 1-level ~8 % vs 2-level ~4 % at 16K-style toy ---- */
    {
        const int S = 4096; /* 64 chunks of 64 */
        hga_config c1 = hga_config_qwen38_27b(1, S, 2);
        c1.n_q_heads = 4; c1.n_kv_heads = 2; c1.head_dim = 32; c1.rotary_dim = 16;
        c1.keep_first = 2; c1.keep_last = 8; c1.theta = 10000.f;
        hga_config c2 = c1; c2.levels = 2;
        const int n_closed = S / c1.chunk_size;
        const int k1 = hga_topk_chunks(&c1, n_closed);
        const int k2 = hga_topk_chunks(&c2, n_closed);
        const int g1 = hga_topk_groups(&c1, n_closed, k1);
        const int g2 = hga_topk_groups(&c2, n_closed, k2);
        std::printf("[budget] n_closed=%d topk_chunks 1L=%d 2L=%d (must match) groups 1L=%d 2L=%d\n",
                    n_closed, k1, k2, g1, g2);
        if (k1 != k2) { ++nfail; std::fprintf(stderr, "chunk counts differ\n"); }
        if (!(g2 * 2 == g1 || std::abs(g1 - 2 * g2) <= 2)) {
            /* 1-level opens all groups in selected chunks; 2-level about half. */
            std::printf("  note: group ratio 1L/2L = %.2f (expect ~2)\n", g1 / (float)std::max(g2, 1));
        }
        if (g2 >= g1 && k1 > 0) { ++nfail; std::fprintf(stderr, "2-level should open fewer groups\n"); }

        hga_session * s1 = hga_session_create(&c1, 1);
        hga_session * s2 = hga_session_create(&c2, 1);
        const int H = c1.n_q_heads, KVH = c1.n_kv_heads, D = c1.head_dim;
        std::vector<float> q((size_t)H * S * D), k((size_t)KVH * S * D), v((size_t)KVH * S * D);
        fill(q, 11); fill(k, 22); fill(v, 33);
        /* Prefill in 64-token blocks, measure last-token sparsity. */
        hga_stats st1{}, st2{};
        const int B = c1.chunk_size;
        for (int p = 0; p < S; p += B) {
            const int n = std::min(B, S - p);
            std::vector<float> qb((size_t)H * n * D), kb((size_t)KVH * n * D), vb((size_t)KVH * n * D);
            std::vector<float> o1((size_t)H * n * D), o2((size_t)H * n * D);
            for (int h = 0; h < H; ++h)
                for (int t = 0; t < n; ++t)
                    for (int d = 0; d < D; ++d)
                        qb[((size_t)h * n + t) * D + d] = q[((size_t)h * S + p + t) * D + d];
            for (int h = 0; h < KVH; ++h)
                for (int t = 0; t < n; ++t)
                    for (int d = 0; d < D; ++d) {
                        kb[((size_t)h * n + t) * D + d] = k[((size_t)h * S + p + t) * D + d];
                        vb[((size_t)h * n + t) * D + d] = v[((size_t)h * S + p + t) * D + d];
                    }
            hga_forward(s1, 0, p, n, qb.data(), kb.data(), kb.data(), vb.data(), HGA_F32, o1.data(), &st1);
            hga_forward(s2, 0, p, n, qb.data(), kb.data(), kb.data(), vb.data(), HGA_F32, o2.data(), &st2);
            for (float x : o1) if (!std::isfinite(x)) { ++nfail; break; }
            for (float x : o2) if (!std::isfinite(x)) { ++nfail; break; }
        }
        std::printf("[sparsity @ %d] 1L chunks=%d groups=%d tokens=%d frac=%.3f | "
                    "2L chunks=%d groups=%d tokens=%d frac=%.3f\n",
                    S, st1.n_selected_chunks, st1.n_opened_groups, st1.n_attended_tokens, st1.sparsity,
                    st2.n_selected_chunks, st2.n_opened_groups, st2.n_attended_tokens, st2.sparsity);
        if (st1.n_selected_chunks != st2.n_selected_chunks) {
            std::fprintf(stderr, "runtime selected-chunk counts differ: %d vs %d\n",
                         st1.n_selected_chunks, st2.n_selected_chunks);
            ++nfail;
        }
        if (st2.n_attended_tokens > st1.n_attended_tokens) {
            std::fprintf(stderr, "2-level attended more tokens than 1-level\n");
            ++nfail;
        }
        hga_session_free(s1);
        hga_session_free(s2);
    }

    /* ---- INT8 KV + dots vs F16 reference (dense window) ---- */
    {
        hga_config cfg = hga_config_qwen38_27b(1, 256, 2);
        cfg.n_q_heads = 8;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 32;
        cfg.rotary_dim = 16;
        cfg.chunk_size = 8;
        cfg.group_size = 4;
        cfg.keep_first = 999;
        cfg.keep_last = 0;
        cfg.frac_l1 = 1.f;
        cfg.theta = 10000.f;
        hga_config cfg8 = cfg;
        cfg.prec = HGA_PREC_F16;
        cfg8.prec = HGA_PREC_I8;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim, S = 24;
        hga_session * sf = hga_session_create(&cfg, 1);
        hga_session * si = hga_session_create(&cfg8, 1);
        std::vector<float> q((size_t)H * S * D), k((size_t)KVH * S * D), v((size_t)KVH * S * D),
            o_f((size_t)H * S * D), o_i((size_t)H * S * D);
        fill(q, 1); fill(k, 2); fill(v, 3);
        for (int t = 0; t < S; ++t) {
            std::vector<float> qt((size_t)H * D), kt((size_t)KVH * D), vt((size_t)KVH * D);
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d)
                    qt[(size_t)h * D + d] = q[((size_t)h * S + t) * D + d];
            for (int h = 0; h < KVH; ++h)
                for (int d = 0; d < D; ++d) {
                    kt[(size_t)h * D + d] = k[((size_t)h * S + t) * D + d];
                    vt[(size_t)h * D + d] = v[((size_t)h * S + t) * D + d];
                }
            std::vector<float> otf((size_t)H * D), oti((size_t)H * D);
            hga_forward(sf, 0, t, 1, qt.data(), kt.data(), kt.data(), vt.data(), HGA_F32, otf.data(), nullptr);
            hga_forward(si, 0, t, 1, qt.data(), kt.data(), kt.data(), vt.data(), HGA_F32, oti.data(), nullptr);
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d) {
                    o_f[((size_t)h * S + t) * D + d] = otf[(size_t)h * D + d];
                    o_i[((size_t)h * S + t) * D + d] = oti[(size_t)h * D + d];
                }
        }
        const float err = max_abs(o_f, o_i);
        double dot = 0, nf = 0, ni = 0;
        for (size_t i = 0; i < o_f.size(); ++i) {
            dot += (double)o_f[i] * (double)o_i[i];
            nf  += (double)o_f[i] * (double)o_f[i];
            ni  += (double)o_i[i] * (double)o_i[i];
        }
        const float cosine = (nf > 0 && ni > 0) ? (float)(dot / (std::sqrt(nf) * std::sqrt(ni))) : 0.f;
        std::printf("[i8_vs_f16] max abs err = %.3e  cosine = %.6f\n", err, cosine);
        if (!(cosine > 0.97f)) {
            std::fprintf(stderr, "  INT8 vs F16 cosine too low\n");
            ++nfail;
        }
        for (float x : o_i) if (!std::isfinite(x)) { ++nfail; break; }
        hga_session_free(sf);
        hga_session_free(si);
    }

    if (nfail) return fail("some tests failed");
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
