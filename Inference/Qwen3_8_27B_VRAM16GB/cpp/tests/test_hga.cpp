#include "hga.h"
#include "hga_l2.h"
#include "hga/profile.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

    {
        constexpr hga::Vram16Qwen38Profile profile{};
        constexpr hga::ModelShape exact{64, 24, 4, 256, 64, 1};
        constexpr hga::ModelShape wrong_heads{64, 23, 4, 256, 64, 1};
        static_assert(profile.validate(exact) == hga::ProfileError::None);
        static_assert(profile.validate(wrong_heads) == hga::ProfileError::HeadCount);
        static_assert(profile.kHgaThreads == 12);
        static_assert(profile.kVerifyKTiles == 3);
        static_assert(profile.kLocalChunks == 7);
        const auto &layers = profile.full_attention_layers();
        const auto &pairs = profile.exchange_pairs();
        int verify_pairs = 0;
        for (const auto &pair : pairs) verify_pairs += pair.streamed_in_verify;
        std::printf("[profile] full_attn=%zu verify_pairs=%d cuda_graphs=%d threads=%u k_tiles=%u local=%u\n",
                    layers.size(), verify_pairs, (int)profile.cuda_graphs_enabled(),
                    profile.kHgaThreads, profile.kVerifyKTiles, profile.kLocalChunks);
        if (layers.front() != 3 || layers.back() != 63 || verify_pairs != 2 ||
            profile.validate(exact) != hga::ProfileError::None ||
            profile.cuda_graphs_enabled() ||
            profile.kHgaThreads != 12 || profile.kVerifyKTiles != 3 ||
            profile.kLocalChunks != 7) {
            std::fprintf(stderr, "  typed production profile mismatch\n"); ++nfail;
        }
    }

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
        c1.keep_first = 2; c1.keep_last = 7; c1.theta = 10000.f;
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
        /* Exclusive frac_l1 at 4K with 10 window chunks: ~5 routed, not 1. */
        if (k1 < 4) {
            std::fprintf(stderr, "  exclusive frac_l1 routed only %d chunks at 4K\n", k1);
            ++nfail;
        }

        /* Query-aware fraction base: a full 8-chunk prefill block absorbs all
         * seven local chunks. Shorter blocks progressively subtract the
         * unused local chunks; verify/decode subtracts all seven. */
        const int base_pf = hga_routing_base_chunks(&c2, n_closed, 512);
        const int base_half = hga_routing_base_chunks(&c2, n_closed, 256);
        const int base_one = hga_routing_base_chunks(&c2, n_closed, 1);
        const int base_two = hga_routing_base_chunks(&c2, n_closed, 2);
        const int base_verify = hga_routing_base_chunks(&c2, n_closed, 3);
        const int k_pf = hga_topk_chunks_for_query(&c2, n_closed, 512);
        const int k_verify = hga_topk_chunks_for_query(&c2, n_closed, 3);
        const int g_pf = hga_topk_groups_for_query(&c2, n_closed, k_pf, 512);
        const int g_verify = hga_topk_groups_for_query(&c2, n_closed, k_verify, 3);
        std::printf("[query_budget] base prefill=%d half=%d verify=%d  topk=%d/%d groups=%d/%d\n",
                    base_pf, base_half, base_verify, k_pf, k_verify, g_pf, g_verify);
        if (base_pf != n_closed - 2 || base_half != n_closed - 6 ||
            base_one != n_closed - 9 || base_two != n_closed - 9 ||
            base_verify != n_closed - 9) {
            std::fprintf(stderr, "  query-aware routing base is wrong\n");
            ++nfail;
        }
        if (k_pf < k_verify || g_pf < g_verify) {
            std::fprintf(stderr, "  full prefill budget must not be smaller than verify\n");
            ++nfail;
        }

        /* Small-context quality floor: use up to three routed middle chunks
         * and six groups, but never exceed what the middle actually holds. */
        const int n_windows = c2.keep_first + c2.keep_last;
        for (int n_available = 0; n_available <= 3; ++n_available) {
            const int closed = n_windows + n_available;
            const int k = hga_topk_chunks_for_query(&c2, closed, 3);
            const int g = hga_topk_groups_for_query(&c2, closed, k, 3);
            const int expected_k = n_available;
            const int expected_g = std::min(6, n_available *
                                               (c2.chunk_size / c2.group_size));
            if (k != expected_k || g != expected_g) {
                std::fprintf(stderr,
                             "  short-context floor closed=%d got=%d/%d expected=%d/%d\n",
                             closed, k, g, expected_k, expected_g);
                ++nfail;
            }
        }

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

    /* ggml [dh, heads, tok] strides + token-major out == packed head-major */
    {
        hga_config cfg = hga_config_qwen38_27b(1, 64, 2);
        cfg.n_q_heads = 4; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.keep_first = 999; cfg.keep_last = 0; cfg.frac_l1 = 1.f;
        cfg.prec = HGA_PREC_I8; cfg.theta = 10000.f;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim, S = 8;
        hga_session * a = hga_session_create(&cfg, 1);
        hga_session * b = hga_session_create(&cfg, 1);
        std::vector<float> q((size_t) H * S * D), k((size_t) KVH * S * D), v((size_t) KVH * S * D);
        fill(q, 7); fill(k, 8); fill(v, 9);
        std::vector<float> qg((size_t) H * S * D), kg((size_t) KVH * S * D), vg((size_t) KVH * S * D);
        auto to_ggml = [](const float * src, float * dst, int heads, int seq, int dim) {
            for (int h = 0; h < heads; ++h)
                for (int t = 0; t < seq; ++t)
                    for (int d = 0; d < dim; ++d)
                        dst[(size_t) t * heads * dim + (size_t) h * dim + d] =
                            src[((size_t) h * seq + t) * dim + d];
        };
        to_ggml(q.data(), qg.data(), H, S, D);
        to_ggml(k.data(), kg.data(), KVH, S, D);
        to_ggml(v.data(), vg.data(), KVH, S, D);
        std::vector<float> oa((size_t) H * S * D), ob((size_t) S * H * D);
        hga_forward(a, 0, 0, S, q.data(), k.data(), k.data(), v.data(), HGA_F32, oa.data(), nullptr);
        hga_forward_strided(b, 0, 0, S,
                            qg.data(), D, H * D,
                            kg.data(), D, KVH * D,
                            kg.data(), D, KVH * D,
                            vg.data(), D, KVH * D,
                            ob.data(), HGA_OUT_TOKEN_MAJOR, nullptr);
        float err = 0.f;
        for (int h = 0; h < H; ++h)
            for (int t = 0; t < S; ++t)
                for (int d = 0; d < D; ++d) {
                    const float x = oa[((size_t) h * S + t) * D + d];
                    const float y = ob[(size_t) t * (H * D) + h * D + d];
                    err = std::max(err, std::fabs(x - y));
                }
        std::printf("[strided_ggml] max abs err vs packed = %.3e\n", err);
        if (!(err < 1e-5f)) {
            std::fprintf(stderr, "  strided path diverged from packed\n");
            ++nfail;
        }
        hga_session_free(a);
        hga_session_free(b);
    }

    /* Spec verify: n_q=3 is 3 decode routings, must match 3 sequential n_q=1. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 64, 2);
        cfg.n_q_heads = 4; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.keep_first = 999; cfg.keep_last = 0; cfg.frac_l1 = 1.f;
        cfg.prec = HGA_PREC_I8; cfg.theta = 10000.f;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim, N = 3;
        hga_session * seq = hga_session_create(&cfg, 1);
        hga_session * bat = hga_session_create(&cfg, 1);
        std::vector<float> q((size_t)H * N * D), k((size_t)KVH * N * D), v((size_t)KVH * N * D);
        fill(q, 3); fill(k, 4); fill(v, 5);
        std::vector<float> oseq((size_t)H * N * D), obat((size_t)H * N * D);
        for (int t = 0; t < N; ++t) {
            std::vector<float> qt((size_t)H * D), kt((size_t)KVH * D), vt((size_t)KVH * D), ot((size_t)H * D);
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d)
                    qt[(size_t)h * D + d] = q[((size_t)h * N + t) * D + d];
            for (int h = 0; h < KVH; ++h)
                for (int d = 0; d < D; ++d) {
                    kt[(size_t)h * D + d] = k[((size_t)h * N + t) * D + d];
                    vt[(size_t)h * D + d] = v[((size_t)h * N + t) * D + d];
                }
            hga_forward(seq, 0, t, 1, qt.data(), kt.data(), kt.data(), vt.data(),
                        HGA_F32, ot.data(), nullptr);
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d)
                    oseq[((size_t)h * N + t) * D + d] = ot[(size_t)h * D + d];
        }
        hga_forward(bat, 0, 0, N, q.data(), k.data(), k.data(), v.data(),
                    HGA_F32, obat.data(), nullptr);
        const float err = max_abs(oseq, obat);
        std::printf("[verify_3x_decode] max abs err vs sequential n_q=1 = %.3e\n", err);
        if (!(err < 2e-3f)) {
            std::fprintf(stderr, "  n_q=3 decode loop diverged from 3x n_q=1\n");
            ++nfail;
        }
        hga_session_free(seq);
        hga_session_free(bat);
    }

    /* Spec verify after a long prefix must preserve the sequential causal
     * result while using one packed final-query K/V list.  This exercises the
     * fused n_q<=8 kernel rather than only the tiny empty-cache case above. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 384, 4);
        cfg.n_q_heads = 8; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.keep_first = 999; cfg.keep_last = 0; cfg.frac_l1 = 1.f;
        cfg.prec = HGA_PREC_I8; cfg.theta = 10000.f;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
        const int P = 192, N = 3;
        hga_session * seq = hga_session_create(&cfg, 1);
        hga_session * bat = hga_session_create(&cfg, 1);
        std::vector<float> qp((size_t)H * P * D), kp((size_t)KVH * P * D), vp((size_t)KVH * P * D);
        std::vector<float> q((size_t)H * N * D), k((size_t)KVH * N * D), v((size_t)KVH * N * D);
        std::vector<float> throwaway((size_t)H * P * D), oseq((size_t)H * N * D), obat((size_t)H * N * D);
        fill(qp, 21); fill(kp, 22); fill(vp, 23);
        fill(q, 24); fill(k, 25); fill(v, 26);
        hga_forward(seq, 0, 0, P, qp.data(), kp.data(), kp.data(), vp.data(), HGA_F32, throwaway.data(), nullptr);
        hga_forward(bat, 0, 0, P, qp.data(), kp.data(), kp.data(), vp.data(), HGA_F32, throwaway.data(), nullptr);
        for (int t = 0; t < N; ++t) {
            std::vector<float> qt((size_t)H * D), kt((size_t)KVH * D), vt((size_t)KVH * D), ot((size_t)H * D);
            for (int h = 0; h < H; ++h)
                std::memcpy(qt.data() + (size_t)h * D,
                            q.data() + ((size_t)h * N + t) * D, (size_t)D * sizeof(float));
            for (int h = 0; h < KVH; ++h) {
                std::memcpy(kt.data() + (size_t)h * D,
                            k.data() + ((size_t)h * N + t) * D, (size_t)D * sizeof(float));
                std::memcpy(vt.data() + (size_t)h * D,
                            v.data() + ((size_t)h * N + t) * D, (size_t)D * sizeof(float));
            }
            hga_forward(seq, 0, P + t, 1, qt.data(), kt.data(), kt.data(), vt.data(),
                        HGA_F32, ot.data(), nullptr);
            for (int h = 0; h < H; ++h)
                std::memcpy(oseq.data() + ((size_t)h * N + t) * D,
                            ot.data() + (size_t)h * D, (size_t)D * sizeof(float));
        }
        hga_forward(bat, 0, P, N, q.data(), k.data(), k.data(), v.data(),
                    HGA_F32, obat.data(), nullptr);
        const float err = max_abs(oseq, obat);
        std::printf("[verify_3x_packed] max abs err vs sequential n_q=1 = %.3e\n", err);
        if (!(err < 2e-3f)) {
            std::fprintf(stderr, "  fused verify diverged after packed-KV prefix\n");
            ++nfail;
        }
        hga_session_free(seq);
        hga_session_free(bat);
    }

    /* Packed L2 tiles: after the window is full, decode should append/reuse not rebuild. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 512, 2);
        cfg.n_q_heads = 8; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.keep_first = 2; cfg.keep_last = 4; cfg.theta = 10000.f;
        cfg.prec = HGA_PREC_I8;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim, S = 192;
        hga_session * sess = hga_session_create(&cfg, 1);
        std::vector<float> q((size_t) H * D), k((size_t) KVH * D), v((size_t) KVH * D), o((size_t) H * D);
        fill(q, 3); fill(k, 4); fill(v, 5);
        hga_stats st{};
        int reuse = 0, append = 0, rebuild = 0;
        for (int t = 0; t < S; ++t) {
            hga_forward(sess, 0, t, 1, q.data(), k.data(), k.data(), v.data(), HGA_F32, o.data(), &st);
            if (t >= S - 8) {
                reuse += st.pack_reuse;
                append += st.pack_append;
                rebuild += st.pack_rebuild;
            }
            for (float x : o) if (!std::isfinite(x)) { ++nfail; break; }
        }
        std::printf("[pack_persist] last8 rebuild=%d append=%d reuse=%d att=%d\n",
                    rebuild, append, reuse, st.n_attended_tokens);
        if (rebuild + append + reuse <= 0) {
            std::fprintf(stderr, "  pack counters idle\n");
            ++nfail;
        }
        if (reuse + append <= rebuild) {
            std::fprintf(stderr, "  packed tiles rebuilt every token; persistence broken\n");
            ++nfail;
        }
        hga_session_free(sess);
    }

    /* last_keys + window keys after a short decode stream */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 256, 2);
        cfg.n_q_heads = 8; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.chunk_size = 8; cfg.group_size = 4;
        cfg.keep_first = 2; cfg.keep_last = 2; cfg.theta = 10000.f;
        cfg.prec = HGA_PREC_I8;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim, S = 40;
        hga_session * sess = hga_session_create(&cfg, 2);
        std::vector<float> q((size_t) H * D), k((size_t) KVH * D), v((size_t) KVH * D), o((size_t) H * D);
        fill(q, 1); fill(k, 2); fill(v, 3);
        for (int t = 0; t < S; ++t) {
            hga_forward(sess, 0, t, 1, q.data(), k.data(), k.data(), v.data(), HGA_F32, o.data(), nullptr);
            hga_forward(sess, 1, t, 1, q.data(), k.data(), k.data(), v.data(), HGA_F32, o.data(), nullptr);
        }
        const int * keys = nullptr;
        int n_keys = 0;
        hga_last_keys(sess, &keys, &n_keys);
        std::printf("[last_keys] n=%d layer=%d n_kv=%d\n", n_keys, hga_last_keys_layer(sess),
                    hga_session_n_kv(sess, 1));
        if (n_keys < 8 || !keys) {
            std::fprintf(stderr, "  last_keys too small\n");
            ++nfail;
        }
        std::vector<int> win(512);
        const int nw = hga_window_keys(sess, 0, S - 1, win.data(), (int) win.size());
        std::printf("[window_keys] n=%d\n", nw);
        if (nw < 8) {
            std::fprintf(stderr, "  window_keys too small\n");
            ++nfail;
        }
        std::vector<int> warm(512);
        const int nwhole = hga_prefetch_keys(
            sess, 1, 1u << 20, warm.data(), (int)warm.size());
        const size_t summary_bytes =
            (size_t)(S / cfg.chunk_size) * (size_t)cfg.n_kv_heads *
            (size_t)cfg.head_dim * sizeof(float) *
            (size_t)(1 + cfg.chunk_size / cfg.group_size);
        const size_t kv_token_bytes =
            2u * (size_t)cfg.n_kv_heads * (size_t)cfg.head_dim +
            2u * (size_t)cfg.n_kv_heads * sizeof(float);
        const int nlimited = hga_prefetch_keys(
            sess, 1, summary_bytes + 24u * kv_token_bytes,
            warm.data(), (int)warm.size());
        std::printf("[prefetch_keys] whole=%d limited=%d first=%d recent=%d\n",
                    nwhole, nlimited, warm[0], warm[nlimited - 1]);
        if (nwhole != S || nlimited != 24 || warm[0] != 0 ||
            warm[15] != 15 || warm[16] != 32 || warm[23] != 39) {
            std::fprintf(stderr, "  prefetch sink/recent budget is wrong\n");
            ++nfail;
        }
        hga_touch_kv_tile(sess, 1, keys, n_keys, 0, 2);
        hga_touch_summary_tile(sess, 1, 0, 2);
        hga_session_free(sess);
    }

    /* Sharded Q4_K GEMV: uniform nibble → y[i] = nibble * d * sum(x) */
    {
        const int n_embd = 512;
        const int n_out  = 64;
        const int n_blk  = n_embd / HGA_QK_K;
        const uint8_t nib = 3;
        const float d = 0.5f;
        std::vector<hga_q4k_block> wk((size_t) n_out * (size_t) n_blk);
        std::vector<hga_q4k_block> wv((size_t) n_out * (size_t) n_blk);
        for (int r = 0; r < n_out; ++r) {
            for (int b = 0; b < n_blk; ++b) {
                hga_q4k_make_uniform(&wk[(size_t) r * n_blk + b], d, nib);
                hga_q4k_make_uniform(&wv[(size_t) r * n_blk + b], d, (uint8_t) (nib + 1));
            }
        }
        std::vector<float> x((size_t) n_embd), k_out((size_t) n_out), v_out((size_t) n_out);
        fill(x, 99);
        float sumx = 0.f;
        for (float v : x) sumx += v;
        const float expect_k = d * (float) nib * sumx;
        const float expect_v = d * (float) (nib + 1) * sumx;

        hga_l2_plan * plan = hga_l2_plan_create(4, 1, n_embd, n_out);
        hga_l2_bind_weights(plan, 0,
                            wk.data(), (size_t) n_blk * sizeof(hga_q4k_block), sizeof(hga_q4k_block),
                            HGA_QK_K, hga_q4k_dequant_block,
                            wv.data(), (size_t) n_blk * sizeof(hga_q4k_block), sizeof(hga_q4k_block),
                            HGA_QK_K, hga_q4k_dequant_block);
        hga_l2_gemv(plan, 0, x.data(), k_out.data(), v_out.data());
        float ek = 0.f, ev = 0.f;
        for (int i = 0; i < n_out; ++i) {
            ek = std::max(ek, std::fabs(k_out[(size_t) i] - expect_k));
            ev = std::max(ev, std::fabs(v_out[(size_t) i] - expect_v));
        }
        std::printf("[l2_gemv_uniform] max abs err K=%.3e V=%.3e  (n_blocks=%d slice0=%zu B)\n",
                    ek, ev, hga_l2_n_blocks(plan), hga_l2_slice_bytes(plan, 0, 0));
        if (!(ek < 1e-3f && ev < 1e-3f)) {
            std::fprintf(stderr, "  sharded GEMV diverged from uniform reference\n");
            ++nfail;
        }

        /* 1-thread must match 4-thread */
        std::vector<float> k1((size_t) n_out), v1((size_t) n_out);
        hga_l2_plan * p1 = hga_l2_plan_create(1, 1, n_embd, n_out);
        hga_l2_bind_weights(p1, 0,
                            wk.data(), (size_t) n_blk * sizeof(hga_q4k_block), sizeof(hga_q4k_block),
                            HGA_QK_K, hga_q4k_dequant_block,
                            wv.data(), (size_t) n_blk * sizeof(hga_q4k_block), sizeof(hga_q4k_block),
                            HGA_QK_K, hga_q4k_dequant_block);
        hga_l2_gemv(p1, 0, x.data(), k1.data(), v1.data());
        float e1 = 0.f;
        for (int i = 0; i < n_out; ++i) {
            e1 = std::max(e1, std::fabs(k1[(size_t) i] - k_out[(size_t) i]));
            e1 = std::max(e1, std::fabs(v1[(size_t) i] - v_out[(size_t) i]));
        }
        std::printf("[l2_gemv_1vs4] max abs err = %.3e\n", e1);
        if (!(e1 < 1e-5f)) {
            std::fprintf(stderr, "  1-thread GEMV != 4-thread\n");
            ++nfail;
        }
        /* Pinned pool path (same math, different threads). */
        hga_l2_plan * pp = hga_l2_plan_create(4, 1, n_embd, n_out);
        if (hga_l2_plan_start(pp) == 4) {
            std::vector<float> kp((size_t) n_out), vp((size_t) n_out);
            hga_l2_bind_weights(pp, 0,
                                wk.data(), (size_t) n_blk * sizeof(hga_q4k_block), sizeof(hga_q4k_block),
                                HGA_QK_K, hga_q4k_dequant_block,
                                wv.data(), (size_t) n_blk * sizeof(hga_q4k_block), sizeof(hga_q4k_block),
                                HGA_QK_K, hga_q4k_dequant_block);
            hga_l2_kick_prefetch(pp, 0, nullptr, nullptr, 0);
            hga_l2_gemv(pp, 0, x.data(), kp.data(), vp.data());
            float ep = 0.f;
            for (int i = 0; i < n_out; ++i) {
                ep = std::max(ep, std::fabs(kp[(size_t) i] - k_out[(size_t) i]));
                ep = std::max(ep, std::fabs(vp[(size_t) i] - v_out[(size_t) i]));
            }
            std::printf("[l2_gemv_pool] max abs err = %.3e\n", ep);
            if (!(ep < 1e-5f)) {
                std::fprintf(stderr, "  pool GEMV != serial\n");
                ++nfail;
            }
        } else {
            std::printf("[l2_gemv_pool] skipped (no pthread pool)\n");
        }
        hga_l2_plan_free(pp);
        hga_l2_plan_free(p1);
        hga_l2_plan_free(plan);
    }

    /* Wave index with retrieval=1 recovers every mid token → last-token ~ dense. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 256, 2);
        cfg.n_q_heads = 4; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.chunk_size = 8; cfg.group_size = 4;
        cfg.keep_first = 2; cfg.keep_last = 2;
        cfg.theta = 10000.f;
        cfg.prec = HGA_PREC_F16;
        cfg.router = HGA_ROUTER_WAVE;
        cfg.frac_retr = 1.f;
        cfg.frac_est = 0.f;
        cfg.wave_cluster = 8;
        cfg.wave_seg = 64;
        cfg.wave_iters = 3;
        cfg.wave_update = 1024;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim, S = 96;
        hga_session * sess = hga_session_create(&cfg, 1);
        std::vector<float> q((size_t)H * S * D), k((size_t)KVH * S * D), v((size_t)KVH * S * D),
            o_hga((size_t)H * S * D), o_ref((size_t)H * S * D);
        fill(q, 4); fill(k, 5); fill(v, 6);
        for (int t = 0; t < S; ++t) {
            std::vector<float> qt((size_t)H * D), kt((size_t)KVH * D), vt((size_t)KVH * D), ot((size_t)H * D);
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d)
                    qt[(size_t)h * D + d] = q[((size_t)h * S + t) * D + d];
            for (int h = 0; h < KVH; ++h)
                for (int d = 0; d < D; ++d) {
                    kt[(size_t)h * D + d] = k[((size_t)h * S + t) * D + d];
                    vt[(size_t)h * D + d] = v[((size_t)h * S + t) * D + d];
                }
            hga_forward(sess, 0, t, 1, qt.data(), kt.data(), kt.data(), vt.data(), HGA_F32, ot.data(), nullptr);
            for (int h = 0; h < H; ++h)
                for (int d = 0; d < D; ++d)
                    o_hga[((size_t)h * S + t) * D + d] = ot[(size_t)h * D + d];
        }
        dense_sdpa(q.data(), k.data(), v.data(), H, KVH, S, D, o_ref.data());
        /* Last query only: earlier tokens may have been hierarchical before the index existed. */
        double dot = 0, na = 0, nb = 0;
        float abserr = 0.f;
        const int t = S - 1;
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d) {
                const float a = o_hga[((size_t)h * S + t) * D + d];
                const float b = o_ref[((size_t)h * S + t) * D + d];
                dot += (double)a * (double)b;
                na += (double)a * (double)a;
                nb += (double)b * (double)b;
                abserr = std::max(abserr, std::fabs(a - b));
            }
        const float cosine = (na > 0 && nb > 0) ? (float)(dot / (std::sqrt(na) * std::sqrt(nb))) : 0.f;
        std::printf("[wave_retr1_vs_dense] last-token cosine=%.6f abserr=%.3e\n", cosine, abserr);
        if (!(cosine > 0.98f)) {
            std::fprintf(stderr, "  wave retr=1 last token diverged from dense\n");
            ++nfail;
        }
        hga_stats st{};
        std::vector<float> qt((size_t)H * D), kt((size_t)KVH * D), vt((size_t)KVH * D), ot((size_t)H * D);
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d)
                qt[(size_t)h * D + d] = q[((size_t)h * S + (S - 1)) * D + d];
        for (int h = 0; h < KVH; ++h)
            for (int d = 0; d < D; ++d) {
                kt[(size_t)h * D + d] = k[((size_t)h * S + (S - 1)) * D + d];
                vt[(size_t)h * D + d] = v[((size_t)h * S + (S - 1)) * D + d];
            }
        /* One more decode step to report stats (index already built). */
        hga_attend(sess, 0, S - 1, 1, qt.data(), HGA_F32, ot.data(), &st);
        std::printf("[wave_stats] n_kv=%d nprobe=%d n_est=%d att_tok=%d frac=%.3f\n",
                    st.n_kv, st.n_selected_chunks, st.n_opened_groups, st.n_attended_tokens, st.sparsity);
        hga_session_free(sess);
    }

    /* Wave with RetroInfer-like 1.8% budget attends fewer tokens than HGA-2 at 1K. */
    {
        const int S = 1024;
        hga_config ch = hga_config_qwen38_27b(2, S + 64, 2);
        ch.n_q_heads = 4; ch.n_kv_heads = 2; ch.head_dim = 32; ch.rotary_dim = 16;
        ch.keep_first = 2; ch.keep_last = 4; ch.theta = 10000.f; ch.prec = HGA_PREC_F16;
        hga_config cw = ch;
        cw.router = HGA_ROUTER_WAVE;
        cw.frac_retr = 0.018f;
        cw.frac_est = 0.232f;
        cw.wave_cluster = 16;
        cw.wave_seg = 256;
        cw.wave_iters = 2;
        const int H = ch.n_q_heads, KVH = ch.n_kv_heads, D = ch.head_dim, B = ch.chunk_size;
        hga_session * sh = hga_session_create(&ch, 1);
        hga_session * sw = hga_session_create(&cw, 1);
        std::vector<float> q((size_t)H * B * D), k((size_t)KVH * B * D), v((size_t)KVH * B * D);
        fill(q, 21); fill(k, 22); fill(v, 23);
        hga_stats sth{}, stw{};
        std::vector<float> o((size_t)H * B * D);
        for (int p = 0; p < S; p += B) {
            const int n = std::min(B, S - p);
            hga_forward(sh, 0, p, n, q.data(), k.data(), k.data(), v.data(), HGA_F32, o.data(), &sth);
            hga_forward(sw, 0, p, n, q.data(), k.data(), k.data(), v.data(), HGA_F32, o.data(), &stw);
        }
        std::vector<float> q1((size_t)H * D), k1((size_t)KVH * D), v1((size_t)KVH * D), o1((size_t)H * D);
        fill(q1, 31); fill(k1, 32); fill(v1, 33);
        hga_forward(sh, 0, S, 1, q1.data(), k1.data(), k1.data(), v1.data(), HGA_F32, o1.data(), &sth);
        hga_forward(sw, 0, S, 1, q1.data(), k1.data(), k1.data(), v1.data(), HGA_F32, o1.data(), &stw);
        std::printf("[wave_vs_hga2] HGA2 att=%d frac=%.3f | wave att=%d frac=%.3f nprobe=%d n_est=%d\n",
                    sth.n_attended_tokens, sth.sparsity, stw.n_attended_tokens, stw.sparsity,
                    stw.n_selected_chunks, stw.n_opened_groups);
        for (float x : o1) if (!std::isfinite(x)) { ++nfail; break; }
        if (stw.n_selected_chunks <= 0) {
            std::fprintf(stderr, "  wave did not retrieve any clusters\n");
            ++nfail;
        }
        hga_session_free(sh);
        hga_session_free(sw);
    }

    /* One-shot append crossing several chunk boundaries must summarize all of
     * them (not only n_kv/C - 1), else later routing sees zero summaries. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 512, 2);
        cfg.n_q_heads = 4; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.chunk_size = 8; cfg.group_size = 4;
        cfg.keep_first = 0; cfg.keep_last = 0;
        cfg.frac_l1 = 0.25f; cfg.frac_l2 = 0.12f;
        cfg.theta = 10000.f; cfg.prec = HGA_PREC_F16;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
        const int C = cfg.chunk_size;
        const int S = 3 * C;
        const int rep = H / KVH;
        hga_session * blocked = hga_session_create(&cfg, 1);
        hga_session * oneshot = hga_session_create(&cfg, 1);
        std::vector<float> q((size_t)H * S * D), k((size_t)KVH * S * D), v((size_t)KVH * S * D);
        fill(q, 41); fill(k, 42); fill(v, 43);
        for (int t = 0; t < C; ++t)
            for (int h = 0; h < KVH; ++h)
                for (int d = 0; d < D; ++d)
                    k[((size_t)h * S + t) * D + d] += 8.f;
        auto slice = [&](int p, int n, std::vector<float> & qb,
                         std::vector<float> & kb, std::vector<float> & vb) {
            qb.assign((size_t)H * n * D, 0.f);
            kb.assign((size_t)KVH * n * D, 0.f);
            vb.assign((size_t)KVH * n * D, 0.f);
            for (int h = 0; h < H; ++h)
                for (int t = 0; t < n; ++t)
                    for (int d = 0; d < D; ++d)
                        qb[((size_t)h * n + t) * D + d] =
                            q[((size_t)h * S + p + t) * D + d];
            for (int h = 0; h < KVH; ++h)
                for (int t = 0; t < n; ++t)
                    for (int d = 0; d < D; ++d) {
                        kb[((size_t)h * n + t) * D + d] =
                            k[((size_t)h * S + p + t) * D + d];
                        vb[((size_t)h * n + t) * D + d] =
                            v[((size_t)h * S + p + t) * D + d];
                    }
        };
        hga_stats stb{}, sto{};
        for (int p = 0; p < S; p += C) {
            std::vector<float> qb, kb, vb, o((size_t)H * C * D);
            slice(p, C, qb, kb, vb);
            hga_forward(blocked, 0, p, C, qb.data(), kb.data(), kb.data(),
                        vb.data(), HGA_F32, o.data(), &stb);
        }
        {
            std::vector<float> o((size_t)H * S * D);
            hga_forward(oneshot, 0, 0, S, q.data(), k.data(), k.data(),
                        v.data(), HGA_F32, o.data(), &sto);
        }
        std::vector<float> q1((size_t)H * D), k1((size_t)KVH * D), v1((size_t)KVH * D);
        std::vector<float> o1((size_t)H * D), o2((size_t)H * D);
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d)
                q1[(size_t)h * D + d] = k[((size_t)(h / rep) * S) * D + d];
        fill(k1, 51); fill(v1, 52);
        hga_forward(blocked, 0, S, 1, q1.data(), k1.data(), k1.data(), v1.data(),
                    HGA_F32, o1.data(), &stb);
        hga_forward(oneshot, 0, S, 1, q1.data(), k1.data(), k1.data(), v1.data(),
                    HGA_F32, o2.data(), &sto);
        const float err = max_abs(o1, o2);
        std::printf("[close_multi_chunk] n_closed block=%d oneshot=%d  last-tok abserr=%.3e  att=%d/%d\n",
                    stb.n_closed_chunks, sto.n_closed_chunks, err,
                    stb.n_attended_tokens, sto.n_attended_tokens);
        if (sto.n_closed_chunks != stb.n_closed_chunks) {
            std::fprintf(stderr, "  closed-chunk count mismatch\n");
            ++nfail;
        }
        if (!(err < 5e-3f)) {
            std::fprintf(stderr, "  oneshot vs blocked diverged\n");
            ++nfail;
        }
        hga_session_free(blocked);
        hga_session_free(oneshot);
    }

    /* Needle in mid-context: exclusive frac should route that chunk, not only
     * sink+local. Distinctive keys in chunk `needle`, query matches them. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 2048, 2);
        cfg.n_q_heads = 4; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.chunk_size = 8; cfg.group_size = 4;
        cfg.keep_first = 2; cfg.keep_last = 4;
        cfg.frac_l1 = 0.08f; cfg.frac_l2 = 0.04f;
        cfg.theta = 10000.f; cfg.prec = HGA_PREC_F16;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
        const int C = cfg.chunk_size;
        const int n_chunks = 16;
        const int S = n_chunks * C;
        const int needle_c = 8;
        const int needle = needle_c * C;
        const int B = C;
        hga_session * sess = hga_session_create(&cfg, 1);
        std::vector<float> q((size_t)H * B * D), k((size_t)KVH * B * D), v((size_t)KVH * B * D);
        std::vector<float> o((size_t)H * B * D);
        fill(q, 61); fill(k, 62); fill(v, 63);
        for (int p = 0; p < S; p += B) {
            if (p == needle) {
                for (float & x : k) x += 6.f;
            } else if (p == needle + B) {
                for (float & x : k) x -= 6.f;
            }
            hga_forward(sess, 0, p, B, q.data(), k.data(), k.data(), v.data(),
                        HGA_F32, o.data(), nullptr);
        }
        std::vector<float> q1((size_t)H * D, 0.f), k1((size_t)KVH * D), v1((size_t)KVH * D),
            o1((size_t)H * D);
        fill(k1, 71); fill(v1, 72);
        /* Query aligned with the boosted chunk-8 keys. */
        for (int h = 0; h < H; ++h)
            for (int d = 0; d < D; ++d)
                q1[(size_t)h * D + d] = 1.f;
        hga_stats st{};
        hga_forward(sess, 0, S, 1, q1.data(), k1.data(), k1.data(), v1.data(),
                    HGA_F32, o1.data(), &st);
        const int * keys = nullptr;
        int n_keys = 0;
        hga_last_keys(sess, &keys, &n_keys);
        bool hit = false;
        for (int i = 0; i < n_keys; ++i)
            if (keys[i] >= needle && keys[i] < needle + C)
                hit = true;
        std::printf("[needle_mid] n_closed=%d selected=%d att=%d hit_chunk8=%d n_keys=%d\n",
                    st.n_closed_chunks, st.n_selected_chunks, st.n_attended_tokens,
                    (int)hit, n_keys);
        if (!hit) {
            std::fprintf(stderr, "  mid-context needle chunk was not attended\n");
            ++nfail;
        }
        hga_session_free(sess);
    }

    /* Fixed VERIFY has physical width three, but padded lanes must be entirely
     * absent from cache state and visible output. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 64, 1);
        cfg.n_q_heads = 4; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.chunk_size = 8; cfg.group_size = 4; cfg.keep_first = 99;
        cfg.keep_last = 0; cfg.frac_l1 = 1.f; cfg.theta = 10000.f;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
        const int P = 3, R = 2;
        hga_session * fixed = hga_session_create(&cfg, 1);
        hga_session * reference = hga_session_create(&cfg, 1);
        std::vector<float> qp((size_t)H * P * D), kp((size_t)KVH * P * D),
            vp((size_t)KVH * P * D), qr((size_t)H * R * D),
            kr((size_t)KVH * R * D), vr((size_t)KVH * R * D),
            op((size_t)H * P * D, -1.f), orf((size_t)H * R * D);
        fill(qp, 81); fill(kp, 82); fill(vp, 83);
        for (int h = 0; h < H; ++h)
            std::memcpy(qr.data() + (size_t)h * R * D,
                        qp.data() + (size_t)h * P * D, (size_t)R * D * sizeof(float));
        for (int h = 0; h < KVH; ++h) {
            std::memcpy(kr.data() + (size_t)h * R * D,
                        kp.data() + (size_t)h * P * D, (size_t)R * D * sizeof(float));
            std::memcpy(vr.data() + (size_t)h * R * D,
                        vp.data() + (size_t)h * P * D, (size_t)R * D * sizeof(float));
        }
        const hga_fixed_batch batch{HGA_MODE_VERIFY, P, R, 0x3};
        if (!hga_forward_fixed(fixed, 0, 0, &batch, qp.data(), kp.data(),
                               kp.data(), vp.data(), HGA_F32, op.data(), nullptr)) {
            std::fprintf(stderr, "  fixed batch rejected\n"); ++nfail;
        }
        hga_forward(reference, 0, 0, R, qr.data(), kr.data(), kr.data(), vr.data(),
                    HGA_F32, orf.data(), nullptr);
        std::vector<float> fixed_real((size_t)H * R * D);
        bool padded_zero = true;
        for (int h = 0; h < H; ++h) {
            std::memcpy(fixed_real.data() + (size_t)h * R * D,
                        op.data() + (size_t)h * P * D, (size_t)R * D * sizeof(float));
            for (int d = 0; d < D; ++d)
                padded_zero = padded_zero && op[((size_t)h * P + R) * D + d] == 0.f;
        }
        const float err = max_abs(fixed_real, orf);
        hga_cache_metrics m{};
        hga_session_cache_metrics(fixed, &m);
        std::printf("[fixed_verify] err=%.3e n_kv=%d appended=%llu\n", err,
                    hga_session_n_kv(fixed, 0),
                    (unsigned long long)m.appended_tokens);
        if (!(err < 2e-3f) || !padded_zero || hga_session_n_kv(fixed, 0) != R ||
            m.appended_tokens != R) {
            std::fprintf(stderr, "  padding changed visible result or cache state\n"); ++nfail;
        }
        if (!hga_truncate(fixed, 0, 1) || hga_session_n_kv(fixed, 0) != 1 ||
            hga_truncate(fixed, 0, 2)) {
            std::fprintf(stderr, "  truncate bounds/state incorrect\n"); ++nfail;
        }
        hga_session_free(fixed);
        hga_session_free(reference);
    }

    /* GPU VERIFY staging must preserve the CPU router's selected-key count,
     * append only real tokens, and encode a fixed-shape causal mask. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 512, 4);
        cfg.n_q_heads = 8; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.chunk_size = 8; cfg.group_size = 4; cfg.keep_first = 1;
        cfg.keep_last = 2; cfg.theta = 10000.f; cfg.prec = HGA_PREC_I8;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
        const int PREFIX = 192, N = 3;
        hga_session * cpu = hga_session_create(&cfg, 1);
        hga_session * gpu = hga_session_create(&cfg, 1);
        std::vector<float> pk((size_t)KVH * PREFIX * D), pv((size_t)KVH * PREFIX * D);
        std::vector<float> q((size_t)H * N * D), k((size_t)KVH * N * D),
            v((size_t)KVH * N * D), out((size_t)H * N * D);
        fill(pk, 81); fill(pv, 82); fill(q, 83); fill(k, 84); fill(v, 85);
        hga_append(cpu, 0, 0, PREFIX, pk.data(), pk.data(), pv.data(), HGA_F32);
        hga_append(gpu, 0, 0, PREFIX, pk.data(), pk.data(), pv.data(), HGA_F32);
        hga_close_full_chunks(cpu, 0);
        hga_close_full_chunks(gpu, 0);
        hga_stats cpu_st{};
        hga_forward(cpu, 0, PREFIX, N, q.data(), k.data(), k.data(), v.data(),
                    HGA_F32, out.data(), &cpu_st);

        const int cap = hga_gpu_verify_capacity(gpu, 0, N);
        hga_gpu_verify_i8_layout layout{};
        if (!hga_gpu_verify_i8_image_layout(gpu, cap, N, &layout)) {
            std::fprintf(stderr, "  verify I8 layout failed\n"); ++nfail;
        }
        std::vector<uint8_t> image(layout.n_bytes, 0xa5u);
        hga_stats gpu_st{};
        const int hist = hga_prepare_gpu_verify_i8_strided(
            gpu, 0, PREFIX, N, N, q.data(), N * D, D,
            k.data(), N * D, D, k.data(), N * D, D,
            v.data(), N * D, D, image.data(), image.size(), cap, &gpu_st);
        const uint16_t zero = 0;
        const uint16_t *mask = (const uint16_t *)(image.data() + layout.mask_offset);
        bool mask_ok = true;
        const int nk = cap + N;
        for (int t = 0; t < N; ++t) {
            for (int x = 0; x < cap; ++x)
                mask_ok = mask_ok && (x < hist
                    ? mask[(size_t)t * nk + x] == zero
                    : mask[(size_t)t * nk + x] != zero);
            for (int x = 0; x < N; ++x)
                mask_ok = mask_ok && (x <= t
                    ? mask[(size_t)t * nk + cap + x] == zero
                    : mask[(size_t)t * nk + cap + x] != zero);
        }
        std::printf("[gpu_verify_stage_i8] cap=%d hist=%d cpu-att=%d gpu-att=%d bytes=%zu mask=%d\n",
                    cap, hist, cpu_st.n_attended_tokens, gpu_st.n_attended_tokens,
                    image.size(), (int)mask_ok);
        if (cap <= 0 || hist < 0 || hist + N != cpu_st.n_attended_tokens ||
            gpu_st.n_attended_tokens != cpu_st.n_attended_tokens || !mask_ok ||
            hga_session_n_kv(gpu, 0) != PREFIX + N) {
            std::fprintf(stderr, "  GPU verify staging diverged from CPU routing\n");
            ++nfail;
        }
        hga_session_free(cpu);
        hga_session_free(gpu);
    }

    /* Hybrid prefill staging advances cache state chunk-by-chunk but emits one
     * compact INT8 historical image for the complete physical ubatch. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 128, 4);
        cfg.n_pack_threads = 2; /* exercise the persistent split packing pool */
        cfg.n_q_heads = 4; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.chunk_size = 8; cfg.group_size = 4; cfg.keep_first = 1;
        cfg.keep_last = 1; cfg.theta = 10000.f; cfg.prec = HGA_PREC_I8;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
        const int N = 16;
        hga_session * sess = hga_session_create(&cfg, 1);
        std::vector<float> q((size_t)H * N * D), k((size_t)KVH * N * D),
            v((size_t)KVH * N * D);
        fill(q, 91); fill(k, 92); fill(v, 93);
        const int need = hga_gpu_prefill_capacity(sess, 0, N);
        const int history_capacity = hga_gpu_prefill_ubatch_history_capacity(
            sess, 0, 0, N, need);
        hga_gpu_prefill_i8_layout layout{};
        if (!hga_gpu_prefill_i8_image_layout(
                sess, history_capacity, N, &layout)) {
            std::fprintf(stderr, "  compact I8 layout failed\n"); ++nfail;
        }
        std::vector<uint8_t> image(layout.n_bytes, 0xa5u);
        hga_stats st{};
        const int got = hga_prepare_gpu_prefill_i8_strided(
            sess, 0, 0, N, q.data(), N * D, D,
            k.data(), N * D, D, k.data(), N * D, D,
            v.data(), N * D, D, image.data(), image.size(), history_capacity, &st);
        bool payload_untouched = true;
        bool scale_zero = true;
        const size_t kv_bytes = (size_t)KVH * history_capacity * D;
        for (size_t x = 0; x < 2 * kv_bytes; ++x)
            payload_untouched = payload_untouched && image[x] == 0xa5u;
        const float *ks = (const float *)(image.data() + layout.k_scale_offset);
        const float *vs = (const float *)(image.data() + layout.v_scale_offset);
        for (int x = 0; x < KVH * history_capacity; ++x)
            scale_zero = scale_zero && ks[x] == 0 && vs[x] == 0;
        hga_cache_metrics metrics{};
        hga_session_cache_metrics(sess, &metrics);
        std::printf("[gpu_prefill_stage_i8] need=%d history=%d bytes=%zu got=%d closed=%d appended=%llu untouched=%d scale0=%d\n",
                    need, history_capacity, image.size(), got, st.n_closed_chunks,
                    (unsigned long long)metrics.appended_tokens,
                    (int)payload_untouched, (int)scale_zero);
        if (need < N || got != N || hga_session_n_kv(sess, 0) != N ||
            st.n_closed_chunks != N / cfg.chunk_size ||
            metrics.appended_tokens != N || !payload_untouched ||
            !scale_zero) {
            std::fprintf(stderr, "  hybrid prefill staging layout/state is wrong\n");
            ++nfail;
        }
        hga_session_free(sess);
    }

    /* Production routing ceilings and the 8K shared-KV union bound. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 8192, 2);
        const int kc = hga_topk_chunks_for_query(&cfg, 4000, cfg.chunk_size);
        const int kg = hga_topk_groups_for_query(&cfg, 4000, kc, cfg.chunk_size);
        hga_session *sess = hga_session_create(&cfg, 1);
        const int cap8k = hga_gpu_prefill_current_capacity(
            sess, 0, 8000, 60);
        const int block0 = hga_gpu_prefill_ubatch_history_capacity(
            sess, 0, 0, 512, 4096);
        const int block8k = hga_gpu_prefill_ubatch_history_capacity(
            sess, 0, 7680, 512, 4096);
        const int block_short = hga_gpu_prefill_ubatch_history_capacity(
            sess, 0, 7680, 312, 4096);
        std::printf("[route_caps] chunks=%d groups=%d cap8k=%d "
                    "block=%d/%d/%d\n",
                    kc, kg, cap8k, block0, block8k, block_short);
        if (kc != 20 || kg != 32 || cap8k <= 0 || cap8k > 1552 ||
            block0 != block8k || block0 != block_short ||
            block0 <= cap8k || block0 > 4096) {
            std::fprintf(stderr, "  production route/union caps are wrong\n");
            ++nfail;
        }
        hga_session_free(sess);
    }

    /* Routes remain independent across query heads and logical chunks, then
     * form one capacity-bounded union without cross-route score comparison. */
    {
        hga_config cfg = hga_config_qwen38_27b(2, 128, 2);
        cfg.n_q_heads = 4; cfg.n_kv_heads = 2; cfg.head_dim = 32; cfg.rotary_dim = 16;
        cfg.chunk_size = 8; cfg.group_size = 4; cfg.keep_first = 0;
        cfg.keep_last = 0; cfg.theta = 10000.f; cfg.prec = HGA_PREC_I8;
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim;
        const int C = cfg.chunk_size, PREFIX = 8 * C, N = 2 * C;
        hga_session * sess = hga_session_create(&cfg, 1);

        std::vector<float> pk((size_t)KVH * PREFIX * D, 0.f);
        std::vector<float> pv((size_t)KVH * PREFIX * D, 0.f);
        for (int kh = 0; kh < KVH; ++kh)
            for (int p = 0; p < PREFIX; ++p) {
                const int cid = p / C;
                pk[((size_t)kh * PREFIX + p) * D + 16 + cid] = 1.f;
                pv[((size_t)kh * PREFIX + p) * D] = (float)(p + 1);
            }
        hga_append(sess, 0, 0, PREFIX, pk.data(), pk.data(), pv.data(), HGA_F32);
        hga_close_full_chunks(sess, 0);

        std::vector<float> q((size_t)H * N * D, 0.f);
        std::vector<float> k((size_t)KVH * N * D, 0.f);
        std::vector<float> v((size_t)KVH * N * D, 0.f);
        for (int kh = 0; kh < KVH; ++kh)
            for (int t = 0; t < N; ++t)
                v[((size_t)kh * N + t) * D] = (float)(PREFIX + t + 1);
        for (int seg = 0; seg < 2; ++seg) {
            for (int t = 0; t < C; ++t) {
                float *q0 = q.data() + ((size_t)0 * N + seg * C + t) * D;
                float *q1 = q.data() + ((size_t)1 * N + seg * C + t) * D;
                if ((t < C / 2) != (seg != 0)) {
                    q0[16] = 10.f; q0[17] = 7.f; q0[18] = 6.f;
                } else {
                    q0[19] = 9.f; q0[20] = 8.f; q0[21] = 5.f;
                }
                q1[17] = 10.f; q1[22] = 9.f; q1[23] = 8.f;
                /* The second KV head is irrelevant to the assertions, but
                 * keep its query heads deterministic and non-degenerate. */
                float *q2 = q.data() + ((size_t)2 * N + seg * C + t) * D;
                float *q3 = q.data() + ((size_t)3 * N + seg * C + t) * D;
                q2[18] = 10.f; q2[20] = 8.f; q2[23] = 7.f;
                q3[19] = 10.f; q3[21] = 8.f; q3[23] = 7.f;
            }
        }

        const int total_cap = hga_gpu_prefill_current_capacity(
            sess, 0, PREFIX, N);
        const int reserved_hist = hga_gpu_prefill_ubatch_history_capacity(
            sess, 0, PREFIX, N, total_cap);
        const int hist_cap = std::min(24, reserved_hist);
        hga_gpu_prefill_i8_layout layout{};
        if (!hga_gpu_prefill_i8_image_layout(
                sess, hist_cap, N, &layout)) {
            std::fprintf(stderr, "  compact route I8 layout failed\n"); ++nfail;
        }
        std::vector<uint8_t> image(layout.n_bytes, 0xa5u);
        hga_stats st{};
        const int got = hga_prepare_gpu_prefill_i8_strided(
            sess, 0, PREFIX, N, q.data(), N * D, D,
            k.data(), N * D, D, k.data(), N * D, D,
            v.data(), N * D, D, image.data(), image.size(), hist_cap, &st);

        bool populated = true;
        const float *kscale =
            (const float *)(image.data() + layout.k_scale_offset);
        const float *vscale =
            (const float *)(image.data() + layout.v_scale_offset);
        for (int kh = 0; kh < KVH; ++kh)
            for (int x = 0; x < hist_cap; ++x)
                populated = populated &&
                    kscale[(size_t)kh * hist_cap + x] != 0.f &&
                    vscale[(size_t)kh * hist_cap + x] != 0.f;
        const bool unioned = st.n_route_group_requests > 0 &&
            st.n_route_group_union > 0 &&
            st.n_route_group_union <= st.n_route_group_requests &&
            st.n_route_group_retained < st.n_route_group_union;
        const bool fanout_ok = st.n_route_group_head_uses >=
                st.n_route_group_union &&
            st.n_route_group_chunk_uses >= st.n_route_group_union &&
            st.n_route_group_max_requests >= st.n_route_group_max_heads &&
            st.n_route_group_max_heads > 0 &&
            st.n_route_group_max_heads <= H / KVH &&
            st.n_route_group_max_chunks > 0 &&
            st.n_route_history_selected > 0 &&
            st.n_route_history_max > 0 &&
            st.n_route_history_max <= hist_cap;
        std::printf("[gpu_prefill_routes] total=%d hist=%d image=%zu got=%d chunks=%d groups=%d requests=%d union=%d retained=%d overlap=%.1f%% fair-topk=%d populated=%d\n",
                    total_cap, hist_cap, image.size(), got,
                    st.n_selected_chunks, st.n_opened_groups,
                    st.n_route_group_requests, st.n_route_group_union,
                    st.n_route_group_retained, 100.f * st.route_group_overlap,
                    st.n_route_topk_limit, (int)populated);
        if (got != hist_cap + N ||
            hga_session_n_kv(sess, 0) != PREFIX + N || !unioned ||
            !fanout_ok || !populated) {
            std::fprintf(stderr, "  united prefill routing/image is wrong\n");
            ++nfail;
        }
        hga_session_free(sess);
    }

    {
        const int S = 512, N = 64, H = 24, D = 256, KVH = 4;
        hga_config cfg = hga_config_qwen38_27b(2, S + 64, 2);
        hga_session *sess = hga_session_create(&cfg, 1);
        std::vector<float> k((size_t)KVH * N * D, 0.1f), v((size_t)KVH * N * D, 0.2f);
        std::vector<float> qp((size_t)N * H * D, 0.3f), qd((size_t)H * D, 0.4f);
        for (int p = 0; p < S; p += N)
            hga_append(sess, 0, p, N, k.data(), k.data(), v.data(), HGA_F32),
                hga_close_full_chunks(sess, 0);
        hga_stats st{};
        hga_route_prefill_only(sess, 0, S - N, N, qp.data(), D, D * H, &st);
        const double pref = st.ms_route;
        hga_route_decode_only(sess, 0, S - 1, qd.data(), D, &st);
        std::printf("[route_only] prefill=%.3f ms decode=%.3f ms kv=%d chunks=%d groups=%d\n",
                    pref, st.ms_route, st.n_kv, st.n_selected_chunks, st.n_opened_groups);
        if (hga_session_n_kv(sess, 0) != S || pref < 0.0 || st.ms_route < 0.0) {
            std::fprintf(stderr, "  route-only API failed\n");
            ++nfail;
        }
        hga_session_free(sess);
    }

    if (nfail) return fail("some tests failed");
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
