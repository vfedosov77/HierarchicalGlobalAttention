#!/usr/bin/env python3
"""Render a self-contained HTML report from an nsys export (.sqlite) + bench log.

Independent (Nsight-Systems) cross-check of the HGA routed-decode bottleneck: raw CUDA
kernels / API / memcpy plus the NVTX per-phase breakdown (emitted with ``HGA_NVTX=1``).

    python profiles/make_report.py profiles/qwen06b_routed_bench.sqlite \
        profiles/bench_run.log profiles/hga_nsys_report.html
"""
from __future__ import annotations

import html
import re
import sqlite3
import sys


def q(c, sql):
    return c.execute(sql).fetchall()


def one(c, sql):
    r = c.execute(sql).fetchone()
    return r if r else (0, 0)


def ms(ns):  # ns -> ms string
    return f"{ns / 1e6:,.2f}"


def bar(pct, color="#4c8bf5"):
    pct = max(0.0, min(100.0, pct))
    return (f'<div class="bar"><div class="fill" style="width:{pct:.1f}%;background:{color}">'
            f'</div><span>{pct:.1f}%</span></div>')


def main(sqlite_path: str, log_path: str, out_path: str) -> None:
    c = sqlite3.connect(sqlite_path)

    # --- GPU busy vs active span ---
    kcount, kbusy = one(c, "SELECT COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL")
    mcount, mtime = one(c, "SELECT COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY")
    span_lo, span_hi = one(c, """SELECT MIN(s), MAX(e) FROM (
        SELECT MIN(start) s, MAX(end) e FROM CUPTI_ACTIVITY_KIND_KERNEL
        UNION ALL SELECT MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_MEMCPY)""")
    span = (span_hi or 0) - (span_lo or 0)
    busy_pct = 100.0 * (kbusy or 0) / span if span else 0.0

    # --- top GPU kernels ---
    kernels = q(c, """SELECT s.value, COUNT(*), SUM(k.end-k.start) t
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        GROUP BY k.shortName ORDER BY t DESC LIMIT 12""")

    # --- CUDA API by time ---
    api = q(c, """SELECT s.value, COUNT(*), SUM(r.end-r.start) t
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        GROUP BY r.nameId ORDER BY t DESC LIMIT 10""")
    api_total = sum(r[2] for r in api) or 1
    launch = next(((n, cnt, t) for n, cnt, t in api if "LaunchKernel" in n), ("cudaLaunchKernel", 0, 0))

    # --- memcpy by direction ---
    memcpy = q(c, """SELECT COALESCE(e.label, 'copyKind='||m.copyKind), COUNT(*),
        SUM(m.end-m.start) t, SUM(m.bytes)
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind
        GROUP BY m.copyKind ORDER BY t DESC""")
    memcpy_total = sum(r[2] for r in memcpy) or 1

    # --- NVTX HGA phases (wall time of push/pop ranges) ---
    nvtx = q(c, """SELECT COALESCE(n.text, s.value) name, COUNT(*), SUM(n.end-n.start) t
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
        WHERE COALESCE(n.text, s.value) LIKE 'hga/%'
        GROUP BY name ORDER BY t DESC""")
    # top-level phases per attention call are mutually-exclusive siblings; their sum is the total
    # attention wall. route_decision/gather_*/assemble are nested INSIDE hga/route.
    top_level = {"hga/qkv_proj", "hga/route", "hga/attend", "hga/o_proj"}
    attn_t = sum(t for n, _, t in nvtx if n in top_level) or 1
    c.close()

    # --- bench numbers from the log ---
    log = open(log_path, encoding="utf-8", errors="replace").read()
    def grab(pat, default="n/a"):
        m = re.search(pat, log)
        return m.group(1) if m else default
    dense = grab(r"original dense\s+([\d.]+) tok/s")
    routed = grab(r"routed RAM\+cache\s+([\d.]+) tok/s")
    ratio = grab(r"speed ratio:\s+([\d.]+)x")
    hitrate = grab(r"hit-rate ([\d.]+)%")

    verdict = "HOST / LAUNCH-BOUND" if busy_pct < 50 else "COMPUTE-BOUND"

    # ---------------- HTML ----------------
    rows_k = "".join(
        f"<tr><td>{ms(t)}</td><td>{cnt:,}</td>"
        f"<td class='pct'>{bar(100*t/(kbusy or 1),'#8e44ad')}</td>"
        f"<td class='name'>{html.escape(n)}</td></tr>"
        for n, cnt, t in kernels)
    rows_api = "".join(
        f"<tr><td>{ms(t)}</td><td>{cnt:,}</td>"
        f"<td class='pct'>{bar(100*t/api_total,'#e67e22')}</td>"
        f"<td class='name'>{html.escape(n)}</td></tr>"
        for n, cnt, t in api)
    rows_mem = "".join(
        f"<tr><td>{ms(t)}</td><td>{cnt:,}</td><td>{(by or 0)/1e6:,.1f} MB</td>"
        f"<td class='pct'>{bar(100*t/memcpy_total,'#16a085')}</td>"
        f"<td class='name'>{html.escape(n)}</td></tr>"
        for n, cnt, t, by in memcpy)
    rows_nvtx = "".join(
        f"<tr><td class='name'>{'&nbsp;&nbsp;&#8627; ' if n not in top_level else ''}"
        f"{html.escape(n)}</td><td>{cnt:,}</td><td>{ms(t)}</td>"
        f"<td class='pct'>{bar(100*t/attn_t, '#c0392b' if n=='hga/attend' else ('#4c8bf5' if n in top_level else '#6b7fae'))}</td></tr>"
        for n, cnt, t in nvtx)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>HGA routed decode — nsys independent profile</title>
<style>
  body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1420;color:#e6e9ef}}
  .wrap{{max-width:1000px;margin:0 auto;padding:28px}}
  h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:28px 0 10px;color:#9db4ff}}
  .sub{{color:#8a93a6;margin-bottom:18px}}
  .cards{{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}}
  .card{{background:#1a2233;border:1px solid #26304a;border-radius:10px;padding:14px 16px;min-width:150px;flex:1}}
  .card .v{{font-size:24px;font-weight:700}} .card .k{{color:#8a93a6;font-size:12px;text-transform:uppercase;letter-spacing:.04em}}
  .verdict{{background:#3a1d1d;border-color:#5c2b2b;color:#ff9d9d}}
  table{{width:100%;border-collapse:collapse;background:#151b29;border-radius:10px;overflow:hidden;font-size:13px}}
  th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #222c42}}
  th{{background:#1c2437;color:#9db4ff;font-weight:600}} td:first-child{{font-variant-numeric:tabular-nums}}
  .name{{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px;color:#c9d3e6}}
  .pct{{width:220px}}
  .bar{{position:relative;background:#222c42;border-radius:5px;height:18px;overflow:hidden}}
  .bar .fill{{position:absolute;left:0;top:0;bottom:0;border-radius:5px}}
  .bar span{{position:absolute;right:6px;top:0;font-size:11px;line-height:18px;color:#e6e9ef}}
  .note{{color:#8a93a6;font-size:12px;margin-top:6px}}
  code{{background:#222c42;padding:1px 5px;border-radius:4px;font-size:12px}}
</style></head><body><div class="wrap">
<h1>Hierarchical Global Attention — routed decode</h1>
<div class="sub">Independent NVIDIA Nsight Systems (nsys) profile &middot; model
<code>Qwen/Qwen3-0.6B</code> &middot; bench: 1024-tok ctx prefill + 32-tok decode, dense &amp; routed</div>

<div class="cards">
  <div class="card verdict"><div class="k">Verdict</div><div class="v">{verdict}</div></div>
  <div class="card"><div class="k">GPU busy</div><div class="v">{busy_pct:.0f}%</div>
     <div class="note">{ms(kbusy)} ms kernels / {ms(span)} ms span</div></div>
  <div class="card"><div class="k">Kernel launches</div><div class="v">{kcount:,}</div>
     <div class="note">{launch[1]:,} via cudaLaunchKernel</div></div>
  <div class="card"><div class="k">Launch API time</div><div class="v">{ms(launch[2])} ms</div>
     <div class="note">&asymp; total kernel time ({ms(kbusy)} ms)</div></div>
</div>
<div class="cards">
  <div class="card"><div class="k">Dense decode</div><div class="v">{dense} <small>tok/s</small></div></div>
  <div class="card"><div class="k">Routed decode</div><div class="v">{routed} <small>tok/s</small></div></div>
  <div class="card"><div class="k">Routed / dense</div><div class="v">{ratio}&times;</div></div>
  <div class="card"><div class="k">Chunk-cache hit</div><div class="v">{hitrate}%</div></div>
</div>

<h2>HGA phase breakdown (NVTX, wall time — routed path only)</h2>
<table><tr><th>Phase</th><th>Instances</th><th>Total (ms)</th><th>% of attention wall</th></tr>
{rows_nvtx}</table>
<div class="note">Top-level phases (<code>qkv_proj</code>, <code>route</code>, <code>attend</code>,
<code>o_proj</code>) are mutually-exclusive siblings that sum to the attention wall; indented rows
(&#8627;) are nested <b>inside</b> <code>hga/route</code>. <b>hga/attend</b> — the real softmax over
token KV (red) — is a small slice, while routing bookkeeping (<code>route_decision</code> top-k +
<code>gather_*</code>) dominates.</div>

<h2>CUDA API (host-side) by time</h2>
<table><tr><th>Total (ms)</th><th>Calls</th><th>% of API</th><th>Name</th></tr>
{rows_api}</table>
<div class="note"><code>cudaLaunchKernel</code> dominates: the host spends roughly as long launching
kernels as the GPU spends running them &rarr; launch-bound.</div>

<h2>GPU kernels by time (top 12)</h2>
<table><tr><th>Total (ms)</th><th>Instances</th><th>% of GPU busy</th><th>Kernel</th></tr>
{rows_k}</table>

<h2>GPU memory copies</h2>
<table><tr><th>Total (ms)</th><th>Count</th><th>Bytes</th><th>% of memcpy</th><th>Direction</th></tr>
{rows_mem}</table>
<div class="note">Transfers are a tiny fraction of the span; H2D is dominated by the one-time
weight/context load, not steady-state decode &rarr; <b>not</b> transfer-bound.</div>

</div></body></html>"""
    open(out_path, "w", encoding="utf-8").write(doc)
    print(f"wrote {out_path}  (GPU busy {busy_pct:.1f}%, {kcount:,} launches, verdict {verdict})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
