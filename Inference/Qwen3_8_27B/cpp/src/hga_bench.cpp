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
    double route_ms;
    double attn_ms;
};

static const char * prec_name(hga_prec p) {
    return p == HGA_PREC_I8 ? "i8" : "f16";
}

static Row run_one(const hga_config & cfg0, int seq, int n_warm) {
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

    Row r{};
    r.seq = seq;
    r.levels = cfg.levels;
    r.topk_chunks = st.n_selected_chunks;
    r.topk_groups = st.n_opened_groups;
    r.att_tokens = st.n_attended_tokens;
    r.frac = st.sparsity;
    r.prefill_ms = t1 - t0;
    r.decode_us = (t3 - t2) * 1e3 / n_rep;
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
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--threads" && i + 1 < argc) n_threads = std::atoi(argv[++i]);
        else if (a == "--max-seq" && i + 1 < argc) max_seq = std::atoi(argv[++i]);
        else if (a == "--tiny") qwen_dims = false;
        else if (a == "--f16") prec = HGA_PREC_F16;
        else if (a == "--i8") prec = HGA_PREC_I8;
        else if (a == "--compare") compare = true;
        else if (a == "-h" || a == "--help") {
            std::printf("hga-bench [--threads N] [--max-seq N] [--tiny] [--i8|--f16] [--compare]\n");
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

    std::printf("# HGA 1-level (8%% tokens, whole 64-token chunks) vs 2-level (4%% tokens, groups of 16)\n");
    std::printf("# Same routed chunk count. threads=%d  %s dims  prec=%s%s\n",
                n_threads, qwen_dims ? "Qwen3.8-27B attn (H=24 KVH=4 Dh=256)" : "tiny",
                prec_name(prec), compare ? " (+compare)" : "");
    std::printf("%8s %6s %5s %8s %8s %9s %7s %12s %12s %10s\n",
                "seq", "level", "prec", "chunks", "groups", "att_tok", "frac", "prefill_ms", "decode_us", "tok/s_dec");
    std::printf("------------------------------------------------------------------------------------------------------\n");

    auto apply_dims = [&](hga_config & c) {
        c.prec = prec;
        if (!qwen_dims) {
            c.n_q_heads = 8; c.n_kv_heads = 2; c.head_dim = 64; c.rotary_dim = 32;
        }
    };

    for (int seq : seqs) {
        Row r1, r2;
        {
            hga_config c = hga_config_qwen38_27b(1, seq + 256, n_threads);
            apply_dims(c);
            r1 = run_one(c, seq, 2);
        }
        {
            hga_config c = hga_config_qwen38_27b(2, seq + 256, n_threads);
            apply_dims(c);
            r2 = run_one(c, seq, 2);
        }
        auto print = [prec](const Row & r) {
            std::printf("%8d %6d %5s %8d %8d %9d %7.3f %12.1f %12.1f %10.1f\n",
                        r.seq, r.levels, prec_name(prec), r.topk_chunks, r.topk_groups, r.att_tokens, r.frac,
                        r.prefill_ms, r.decode_us, 1e6 / std::max(r.decode_us, 1e-6));
        };
        print(r1);
        print(r2);
        const double speedup = r1.decode_us / std::max(r2.decode_us, 1e-9);
        std::printf("         2-level decode speedup vs 1-level: %.2fx   prefill: %.2fx\n",
                    speedup, r1.prefill_ms / std::max(r2.prefill_ms, 1e-9));
        if (compare) {
            hga_config c1 = hga_config_qwen38_27b(2, seq + 256, n_threads);
            apply_dims(c1);
            c1.prec = (prec == HGA_PREC_I8) ? HGA_PREC_F16 : HGA_PREC_I8;
            Row ro = run_one(c1, seq, 2);
            std::printf("%8d %6d %5s %8d %8d %9d %7.3f %12.1f %12.1f %10.1f\n",
                        ro.seq, ro.levels, prec_name(c1.prec), ro.topk_chunks, ro.topk_groups, ro.att_tokens, ro.frac,
                        ro.prefill_ms, ro.decode_us, 1e6 / std::max(ro.decode_us, 1e-6));
            std::printf("         2-level decode time %s/%s = %.2f   prefill = %.2f  (<1 means %s faster)\n",
                        prec_name(prec), prec_name(c1.prec),
                        r2.decode_us / std::max(ro.decode_us, 1e-9),
                        r2.prefill_ms / std::max(ro.prefill_ms, 1e-9),
                        prec_name(prec));
        }
    }
    return 0;
}
