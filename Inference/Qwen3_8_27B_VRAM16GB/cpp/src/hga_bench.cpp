#include "hga.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

static float urand(uint32_t & s) {
    s = s * 1664525u + 1013904223u;
    return (float)((s >> 8) & 0xffffff) / 16777216.f * 2.f - 1.f;
}

static double now_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

struct Row {
    int seq;
    int levels;
    int topk_chunks;
    int topk_groups;
    int att_tokens;
    float frac;
    double prefill_ms;
    double decode_us;
    double verify_us;
    double route_ms;
    double attn_ms;
};

static const char * prec_name(hga_prec p) {
    return p == HGA_PREC_I8 ? "i8" : "f16";
}

static Row run_one(const hga_config & cfg0, int seq, int n_warm, int n_verify = 0) {
    hga_config cfg = cfg0;
    cfg.max_seq = seq + cfg.chunk_size;
    hga_session * s = hga_session_create(&cfg, 1);
    const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim, C = cfg.chunk_size;
    uint32_t seed = 7;
    std::vector<float> q((size_t)H * C * D), k((size_t)KVH * C * D), v((size_t)KVH * C * D),
        o((size_t)H * C * D);
    for (float & x : q) x = urand(seed);
    for (float & x : k) x = urand(seed);
    for (float & x : v) x = urand(seed);

    hga_stats st{};
    const double t0 = now_ms();
    for (int p = 0; p < seq; p += C) {
        const int n = std::min(C, seq - p);
        hga_forward(s, 0, p, n, q.data(), k.data(), k.data(), v.data(), HGA_F32, o.data(), &st);
    }
    const double t1 = now_ms();

    /* Decode: one new token, repeat. */
    std::vector<float> q1((size_t)H * D), k1((size_t)KVH * D), v1((size_t)KVH * D), o1((size_t)H * D);
    std::memcpy(q1.data(), q.data(), q1.size() * sizeof(float));
    std::memcpy(k1.data(), k.data(), k1.size() * sizeof(float));
    std::memcpy(v1.data(), v.data(), v1.size() * sizeof(float));
    for (int i = 0; i < n_warm; ++i) {
        hga_forward(s, 0, seq + i, 1, q1.data(), k1.data(), k1.data(), v1.data(), HGA_F32, o1.data(), &st);
    }
    const int n_rep = 16;
    const double t2 = now_ms();
    for (int i = 0; i < n_rep; ++i) {
        hga_forward(s, 0, seq + n_warm + i, 1, q1.data(), k1.data(), k1.data(), v1.data(), HGA_F32, o1.data(), &st);
    }
    const double t3 = now_ms();

    /* Spec verify uses a short causal batch.  Pack the repeated single-token
     * inputs into HGA's [head, token, dim] layout rather than passing the
     * 64-token prefill buffers directly. */
    double verify_us = 0;
    if (n_verify > 0) {
        std::vector<float> qv((size_t)H * n_verify * D),
            kv((size_t)KVH * n_verify * D), vv((size_t)KVH * n_verify * D),
            ov((size_t)H * n_verify * D);
        for (int h = 0; h < H; ++h)
            for (int t = 0; t < n_verify; ++t)
                std::memcpy(qv.data() + ((size_t)h * n_verify + t) * D,
                            q1.data() + (size_t)h * D, (size_t)D * sizeof(float));
        for (int h = 0; h < KVH; ++h)
            for (int t = 0; t < n_verify; ++t) {
                std::memcpy(kv.data() + ((size_t)h * n_verify + t) * D,
                            k1.data() + (size_t)h * D, (size_t)D * sizeof(float));
                std::memcpy(vv.data() + ((size_t)h * n_verify + t) * D,
                            v1.data() + (size_t)h * D, (size_t)D * sizeof(float));
            }
        int pos = seq + n_warm + n_rep;
        for (int i = 0; i < 4; ++i, pos += n_verify)
            hga_forward(s, 0, pos, n_verify, qv.data(), kv.data(), kv.data(),
                        vv.data(), HGA_F32, ov.data(), &st);
        const int n_verify_rep = 16;
        const double t4 = now_ms();
        for (int i = 0; i < n_verify_rep; ++i, pos += n_verify)
            hga_forward(s, 0, pos, n_verify, qv.data(), kv.data(), kv.data(),
                        vv.data(), HGA_F32, ov.data(), &st);
        const double t5 = now_ms();
        verify_us = (t5 - t4) * 1e3 / n_verify_rep;
    }

    Row r{};
    r.seq = seq;
    r.levels = cfg.levels;
    r.topk_chunks = st.n_selected_chunks;
    r.topk_groups = st.n_opened_groups;
    r.att_tokens = st.n_attended_tokens;
    r.frac = st.sparsity;
    r.prefill_ms = t1 - t0;
    r.decode_us = (t3 - t2) * 1e3 / n_rep;
    r.verify_us = verify_us;
    r.route_ms = st.ms_route;
    r.attn_ms = st.ms_attn;
    hga_session_free(s);
    return r;
}

int main(int argc, char ** argv) {
    int n_threads = 0;
    int max_seq = 32768;
    bool qwen_dims = true;
    hga_prec prec = HGA_PREC_I8;
    bool compare = false;
    bool wave = false;
    bool quality = false;
    bool verify = false;
    int verify_width = 3;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) n_threads = std::atoi(argv[++i]);
        else if (a == "--max-seq" && i + 1 < argc) max_seq = std::atoi(argv[++i]);
        else if (a == "--tiny") qwen_dims = false;
        else if (a == "--f16") prec = HGA_PREC_F16;
        else if (a == "--i8") prec = HGA_PREC_I8;
        else if (a == "--compare") compare = true;
        else if (a == "--wave") wave = true;
        else if (a == "--quality") quality = true;
        else if (a == "--verify") verify = true;
        else if (a == "--verify-width" && i + 1 < argc) {
            verify = true;
            verify_width = std::max(1, std::min(8, std::atoi(argv[++i])));
        }
        else if (a == "-h" || a == "--help") {
            std::printf("hga-bench [--threads N] [--max-seq N] [--tiny] [--i8|--f16] [--compare] [--wave] [--quality] [--verify] [--verify-width 1..8]\n");
            return 0;
        }
    }
    if (n_threads <= 0) {
#if defined(_OPENMP)
        n_threads = omp_get_max_threads();
#else
        n_threads = 8;
#endif
    }

    std::vector<int> seqs = {4096, 8192, 16384, 32768};
    while (!seqs.empty() && seqs.back() > max_seq) seqs.pop_back();
    if (seqs.empty()) seqs.push_back(max_seq);

    if (wave) {
        std::printf("# HGA 2-level vs RetroInfer-style WAVE (retr=1.8%%, est=23.2%%, cluster=16)\n");
    } else {
        std::printf("# HGA 1-level (8%% tokens, whole 64-token chunks) vs 2-level (4%% tokens, groups of 16)\n");
    }
    std::printf("# threads=%d  %s dims  prec=%s%s%s\n",
                n_threads, qwen_dims ? "Qwen3.8-27B attn (H=24 KVH=4 Dh=256)" : "tiny",
                prec_name(prec), compare ? " (+compare)" : "", quality ? " (+quality)" : "");
    if (verify) {
        std::printf("%8s %6s %5s %8s %8s %9s %7s %12s %12s %12s %10s\n",
                    "seq", "mode", "prec", "chunks", "groups", "att_tok", "frac", "prefill_ms", "decode_us", "verify_us", "tok/s_dec");
        std::printf("-------------------------------------------------------------------------------------------------------------------\n");
    } else {
        std::printf("%8s %6s %5s %8s %8s %9s %7s %12s %12s %10s\n",
                    "seq", "mode", "prec", "chunks", "groups", "att_tok", "frac", "prefill_ms", "decode_us", "tok/s_dec");
        std::printf("------------------------------------------------------------------------------------------------------\n");
    }

    auto apply_dims = [&](hga_config & c) {
        c.prec = prec;
        if (!qwen_dims) {
            c.n_q_heads = 8; c.n_kv_heads = 2; c.head_dim = 64; c.rotary_dim = 32;
        }
    };

    auto last_token_cosine = [&](const hga_config & cfg0, int seq) -> float {
        /* Dense last-query vs sparse last-query. Prefill uses the same router. */
        hga_config cfg = cfg0;
        cfg.max_seq = seq + cfg.chunk_size;
        hga_config dense = cfg;
        dense.keep_first = 999;
        dense.keep_last = 0;
        dense.frac_l1 = 1.f;
        dense.router = HGA_ROUTER_HIER;
        hga_session * sp = hga_session_create(&cfg, 1);
        hga_session * sd = hga_session_create(&dense, 1);
        const int H = cfg.n_q_heads, KVH = cfg.n_kv_heads, D = cfg.head_dim, C = cfg.chunk_size;
        std::vector<float> q((size_t)H * C * D), k((size_t)KVH * C * D), v((size_t)KVH * C * D),
            o((size_t)H * C * D);
        for (int p = 0; p < seq; p += C) {
            const int n = std::min(C, seq - p);
            uint32_t seed = 11u + (uint32_t)p * 17u;
            for (float & x : q) x = urand(seed);
            for (float & x : k) x = urand(seed);
            for (float & x : v) x = urand(seed);
            hga_forward(sp, 0, p, n, q.data(), k.data(), k.data(), v.data(), HGA_F32, o.data(), nullptr);
            hga_forward(sd, 0, p, n, q.data(), k.data(), k.data(), v.data(), HGA_F32, o.data(), nullptr);
        }
        std::vector<float> q1((size_t)H * D), k1((size_t)KVH * D), v1((size_t)KVH * D);
        std::vector<float> op((size_t)H * D), od((size_t)H * D);
        std::memcpy(q1.data(), q.data(), q1.size() * sizeof(float));
        std::memcpy(k1.data(), k.data(), k1.size() * sizeof(float));
        std::memcpy(v1.data(), v.data(), v1.size() * sizeof(float));
        hga_forward(sp, 0, seq, 1, q1.data(), k1.data(), k1.data(), v1.data(), HGA_F32, op.data(), nullptr);
        hga_forward(sd, 0, seq, 1, q1.data(), k1.data(), k1.data(), v1.data(), HGA_F32, od.data(), nullptr);
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < op.size(); ++i) {
            dot += (double)op[i] * (double)od[i];
            na += (double)op[i] * (double)op[i];
            nb += (double)od[i] * (double)od[i];
        }
        hga_session_free(sp);
        hga_session_free(sd);
        if (na <= 0 || nb <= 0) return 0.f;
        return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
    };

    for (int seq : seqs) {
        Row r1, r2;
        if (wave) {
            hga_config c2 = hga_config_qwen38_27b(2, seq + 256, n_threads);
            apply_dims(c2);
            r1 = run_one(c2, seq, 2, verify ? verify_width : 0);
            hga_config cw = c2;
            cw.router = HGA_ROUTER_WAVE;
            r2 = run_one(cw, seq, 2, verify ? verify_width : 0);
        } else {
            hga_config c = hga_config_qwen38_27b(1, seq + 256, n_threads);
            apply_dims(c);
            r1 = run_one(c, seq, 2, verify ? verify_width : 0);
            hga_config c2 = hga_config_qwen38_27b(2, seq + 256, n_threads);
            apply_dims(c2);
            r2 = run_one(c2, seq, 2, verify ? verify_width : 0);
        }
        auto print = [prec, verify](const Row & r, const char * tag) {
            if (verify) {
                std::printf("%8d %6s %5s %8d %8d %9d %7.3f %12.1f %12.1f %12.1f %10.1f\n",
                            r.seq, tag, prec_name(prec), r.topk_chunks, r.topk_groups, r.att_tokens, r.frac,
                            r.prefill_ms, r.decode_us, r.verify_us, 1e6 / std::max(r.decode_us, 1e-6));
            } else {
                std::printf("%8d %6s %5s %8d %8d %9d %7.3f %12.1f %12.1f %10.1f\n",
                            r.seq, tag, prec_name(prec), r.topk_chunks, r.topk_groups, r.att_tokens, r.frac,
                            r.prefill_ms, r.decode_us, 1e6 / std::max(r.decode_us, 1e-6));
            }
        };
        if (wave) {
            print(r1, "hga2");
            print(r2, "wave");
            std::printf("         wave decode vs HGA-2: %.2fx   prefill: %.2fx  (>1 = wave faster)\n",
                        r1.decode_us / std::max(r2.decode_us, 1e-9),
                        r1.prefill_ms / std::max(r2.prefill_ms, 1e-9));
        } else {
            print(r1, "1");
            print(r2, "2");
            const double speedup = r1.decode_us / std::max(r2.decode_us, 1e-9);
            std::printf("         2-level decode speedup vs 1-level: %.2fx   prefill: %.2fx\n",
                        speedup, r1.prefill_ms / std::max(r2.prefill_ms, 1e-9));
        }
        if (quality && seq <= 8192) {
            hga_config cq = hga_config_qwen38_27b(2, seq + 256, n_threads);
            apply_dims(cq);
            const float c_hga = last_token_cosine(cq, seq);
            float c_wave = 0.f;
            if (wave) {
                cq.router = HGA_ROUTER_WAVE;
                c_wave = last_token_cosine(cq, seq);
            }
            if (wave)
                std::printf("         last-token cosine vs dense window: HGA-2=%.5f  wave=%.5f\n",
                            c_hga, c_wave);
            else
                std::printf("         last-token cosine vs dense window: HGA-2=%.5f\n", c_hga);
        }
        if (compare && !wave) {
            hga_config c1 = hga_config_qwen38_27b(2, seq + 256, n_threads);
            apply_dims(c1);
            c1.prec = (prec == HGA_PREC_I8) ? HGA_PREC_F16 : HGA_PREC_I8;
            Row ro = run_one(c1, seq, 2, verify ? verify_width : 0);
            print(ro, "2");
            std::printf("         2-level decode time %s/%s = %.2f   prefill = %.2f  (<1 means %s faster)\n",
                        prec_name(prec), prec_name(c1.prec),
                        r2.decode_us / std::max(ro.decode_us, 1e-9),
                        r2.prefill_ms / std::max(ro.prefill_ms, 1e-9),
                        prec_name(prec));
        }
    }
    return 0;
}
