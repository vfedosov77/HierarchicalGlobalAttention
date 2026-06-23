#!/usr/bin/env python3
"""Interactive chat with Qwen3-30B-A3B-Instruct-2507-FP8 on a **RAM-cached KvRouter**.

The attention of every layer is replaced by ``QwenRoutedAttention`` (exact mode) backed by a
``RamKVCacheStore``: the full KV cache lives in host RAM and only the routed chunks (a few sink
+ local + top-k middle chunks) are pulled to VRAM each step.  VRAM use is therefore bounded by
the model weights regardless of context length, so this fits long histories on a 32GB card.

Usage:
    source ~/my_env/bin/activate
    cd ~/HierarchicalGlobalAttention

    # Terminal chat (default):
    python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen30b_fp8

    # Browser UI (auto-opens http://127.0.0.1:7860):
    python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen30b_fp8 --ui
    python -m ExistingModelFineTuning.Qwen3LongContext.chat_qwen30b_fp8 --ui --port 8080
"""

from __future__ import annotations

import os

# Set before CUDA initialises: avoids the FP8 Triton matmul autotuner OOMing in the small VRAM
# headroom left after the ~29GB of weights, and reduces allocator fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import queue
import socketserver
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Generator, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from ExistingModelFineTuning.Qwen3LongContext.qwen_routed_attention import (
    replace_qwen_attention_with_router,
)


MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
MAX_NEW_TOKENS = 32 * 1024

# --- RAM-cached router config (chunk_size 64) ---
# Group-level routing (the active config): a wide chunk pool (TOPK_CHUNKS) routed at group
# granularity, of which each query opens TOPK_GROUPS//2 groups (the implemented per-query logic).
# At 4K this matched the dense baseline (99.76% greedy, ppl 1.262 vs 1.261; 100% in the routing tail)
# while attending only TOPK_GROUPS//2 * GROUP_SIZE = 256 middle tok/step — half the whole-chunk
# cost, so it decodes faster.  For whole-chunk routing instead, set GROUP_SIZE=64, TOPK_GROUPS=2*TOPK_CHUNKS.
CHUNK_SIZE = 64
GROUP_SIZE = 16      # routing granularity: < CHUNK_SIZE = group-level; == CHUNK_SIZE = whole-chunk
KEEP_FIRST = 2       # always-resident leading chunks (attention sinks): 128 tokens
KEEP_LAST = 8        # always-resident trailing chunks (local context): 512 tokens
TOPK_CHUNKS = 20     # routed middle chunks in the candidate pool
TOPK_GROUPS = 32     # groups materialized per step; each query opens TOPK_GROUPS//2 = 16 (256 tok)
# Prefill is fed in blocks of this many tokens.  Kept modest on the 30B FP8: the fp8 matmul
# autotuner OOMs on a large prefill matmul in the ~3GB free after the weights.  One chunk per
# block bounds the activation peak; the VRAM chunk bank auto-sizes to whatever free VRAM is left
# (often ~0 here, so it stays off and VRAM is bounded by the weights — see VRAM_CACHE_* below).
PREFILL_BLOCK = 64
# Upper bound for the LRU VRAM cache of chunk KV; the store auto-shrinks it to fit free VRAM
# (leaving VRAM_CACHE_RESERVE_GB for activations), so a long-context prefill never OOMs the bank.
VRAM_CACHE_CHUNKS = 400
VRAM_CACHE_RESERVE_GB = 1.5
# Independent VRAM cache for group **summaries** (M·Dh per chunk ≈ C/M = 16× smaller than a token
# chunk).  Group-level routing only reads summaries, so caching many of them keeps the per-step
# routing decision GPU-resident and stops it from dragging whole token chunks across PCIe just to
# score them.  Sized to span a long context (≈ chunks at 32K with chunk_size 64) so it sees ≈0
# misses; auto-shrinks to free VRAM (it is tiny, so it almost always fits in full).
VRAM_SUMMARY_CHUNKS = 8192
# Cold-KV tier: "ram" keeps the whole KV record in host RAM; "fs" makes host RAM a bounded LRU page
# cache (RAM_BUDGET_GB) backed by NVMe/disk spillover, so contexts larger than RAM stay on disk
# instead of exhausting memory.  Disk files are removed on exit / Ctrl-C / reset (never left behind),
# and explicit pread/pwrite + posix_fadvise keep the OS responsive (it never behaves like swap).
# Default "fs": host RAM is bounded and the cold KV spills to disk, so long contexts never OOM RAM.
CACHE_LOCATION = "fs"
# Host-RAM ceiling for the "fs" tier (the bulk KV record across all layers).  Beyond this, the
# least-recently-used chunks spill to disk.  Ignored when CACHE_LOCATION != "fs".
RAM_BUDGET_GB = 6.0
# Where spilled chunks live.  MUST be a real disk (NVMe/SSD), never a tmpfs like /tmp or /dev/shm
# (those are RAM-backed and would put the "disk" tier back in RAM).  Default: the user's XDG cache
# dir ($XDG_CACHE_HOME or ~/.cache) — on a real disk for virtually every Linux install, never tmpfs,
# user-writable without root, and survives across runs.  Override with --fs-cache-dir or
# $KVR_FS_CACHE_DIR (e.g. a fast NVMe scratch mount).
FS_CACHE_DIR = os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "kvr_fscache"
)
# Dual Chunk Attention: remaps key/query RoPE positions so every relative position stays inside the
# pretrained window, letting the context grow far beyond max_position_embeddings.  DCA_CHUNK is the
# DCA chunk length (must be a multiple of CHUNK_SIZE); set it below the pretrained window (e.g.
# ~3/4 of it).  0 disables DCA (exact absolute-RoPE behavior).  DCA_LOCAL is the local window added
# on top (defaults to DCA_CHUNK // 5 when 0).

DCA_CHUNK = 131072   # = chunk_size
DCA_LOCAL = 4096     # - a strange clamp of the curent chunk in current Qwen DCA - should be removed.


# ---------------------------------------------------------------------------
# HTML for the browser UI (no external dependencies)
# ---------------------------------------------------------------------------

_CHAT_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen3-30B Chat</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#f0f2f5;height:100dvh;display:flex;flex-direction:column;overflow:hidden}
#hdr{background:#1a1a2e;color:#fff;padding:10px 18px;display:flex;
  align-items:center;justify-content:space-between;flex-shrink:0;gap:10px}
#hdr h1{font-size:.9rem;font-weight:600;letter-spacing:.02em;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
#ctrl{display:flex;gap:7px;flex-shrink:0}
.btn{cursor:pointer;border:none;border-radius:6px;padding:5px 12px;
  font-size:.78rem;font-weight:500;transition:opacity .15s}
.btn:hover{opacity:.8}
#btn-think{background:#2ecc71;color:#fff}
#btn-think.on{background:#e67e22}
#btn-reset{background:#e74c3c;color:#fff}
#msgs{flex:1;overflow-y:auto;padding:14px 18px;display:flex;flex-direction:column;gap:10px}
.bubble{max-width:80%;padding:9px 13px;border-radius:14px;
  line-height:1.6;white-space:pre-wrap;word-break:break-word;font-size:.88rem}
.user{background:#0084ff;color:#fff;align-self:flex-end;border-bottom-right-radius:3px}
.asst{background:#fff;color:#111;align-self:flex-start;
  border-bottom-left-radius:3px;box-shadow:0 1px 4px rgba(0,0,0,.1)}
.asst.gen::after{content:'▋';animation:blink .6s step-end infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
.stat{font-size:.71rem;color:#aaa;align-self:flex-end;margin-top:-4px;padding-right:2px}
#hint{font-size:.71rem;color:#bbb;text-align:center;padding:3px 0 5px;
  flex-shrink:0;background:#f0f2f5}
#inp{background:#fff;border-top:1px solid #e0e0e0;padding:11px 16px;
  display:flex;gap:9px;align-items:flex-end;flex-shrink:0}
#txt{flex:1;border:1.5px solid #d0d0d0;border-radius:10px;padding:9px 13px;
  font:inherit;font-size:.88rem;resize:none;min-height:40px;max-height:200px;
  overflow-y:auto;line-height:1.6;outline:none;transition:border-color .15s}
#txt:focus{border-color:#0084ff}
#btn-send{background:#0084ff;color:#fff;border-radius:10px;padding:9px 18px;
  font-size:.88rem;height:40px;cursor:pointer;border:none;font-weight:500;
  white-space:nowrap;transition:opacity .15s}
#btn-send:disabled{background:#ccc;cursor:default}
</style>
</head>
<body>
<div id="hdr">
  <h1>Qwen3-30B &middot; RAM-cached KvRouter</h1>
  <div id="ctrl">
    <button class="btn" id="btn-think">Thinking: OFF</button>
    <button class="btn" id="btn-reset">Reset</button>
  </div>
</div>
<div id="msgs"></div>
<div id="hint">Enter = new line &nbsp;&middot;&nbsp; Ctrl+Enter = send</div>
<div id="inp">
  <textarea id="txt" rows="1" placeholder="Type a message…"></textarea>
  <button id="btn-send">Send</button>
</div>
<script>
const msgs=document.getElementById('msgs'),
      txt=document.getElementById('txt'),
      btnSend=document.getElementById('btn-send'),
      btnThink=document.getElementById('btn-think'),
      btnReset=document.getElementById('btn-reset');
let thinking=false,busy=false;

txt.addEventListener('input',()=>{
  txt.style.height='auto';
  txt.style.height=Math.min(txt.scrollHeight,200)+'px';
});

function bubble(cls,text){
  const d=document.createElement('div');
  d.className='bubble '+cls;
  if(text)d.textContent=text;
  msgs.appendChild(d);
  msgs.scrollTop=msgs.scrollHeight;
  return d;
}
function stat(s){
  const d=document.createElement('div');
  d.className='stat';d.textContent=s;
  msgs.appendChild(d);
}

async function send(){
  const text=txt.value;
  if(!text.trim()||busy)return;
  busy=true;btnSend.disabled=true;
  txt.value='';txt.style.height='auto';

  bubble('user',text);
  const aDiv=bubble('asst');
  aDiv.classList.add('gen');

  try{
    const resp=await fetch('/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text,thinking})
    });
    if(!resp.ok){aDiv.textContent='[HTTP '+resp.status+']';return;}
    const reader=resp.body.getReader(),dec=new TextDecoder();
    let buf='';
    outer:while(true){
      const{value,done}=await reader.read();
      if(done)break;
      buf+=dec.decode(value,{stream:true});
      const parts=buf.split('\\n\\n');buf=parts.pop();
      for(const p of parts){
        if(!p.startsWith('data:'))continue;
        const raw=p.slice(5).trim();
        if(raw==='[DONE]')break outer;
        try{
          const obj=JSON.parse(raw);
          if(obj.text!==undefined){
            aDiv.textContent+=obj.text;
            msgs.scrollTop=msgs.scrollHeight;
          }else if(obj.stats){stat(obj.stats);}
          else if(obj.error){aDiv.textContent+='\\n[Error: '+obj.error+']';}
        }catch{}
      }
    }
  }catch(e){aDiv.textContent='['+e+']';}
  finally{
    aDiv.classList.remove('gen');
    busy=false;btnSend.disabled=false;
    txt.focus();
  }
}

btnSend.addEventListener('click',send);
txt.addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.key==='Enter'){e.preventDefault();send();}
});
btnThink.addEventListener('click',()=>{
  thinking=!thinking;
  btnThink.textContent='Thinking: '+(thinking?'ON':'OFF');
  btnThink.classList.toggle('on',thinking);
});
btnReset.addEventListener('click',async()=>{
  await fetch('/reset',{method:'POST'});
  msgs.innerHTML='';
});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Core generation (shared by terminal and UI modes)
# ---------------------------------------------------------------------------

def gb(x: int) -> float:
    return x / 1024**3


def _plan_prefill(cache, cached_ids: list[int], new_ids: list[int]):
    """Decide how to seed the next turn's prefill.

    Multi-turn chat is append-only: the new prompt (full chat template + generation prompt) is
    almost always the previous prompt **plus** the last reply and the new user turn, i.e. an
    extension of what the persistent KV cache already holds.  When that prefix relation holds we
    keep the existing cache/router/store and prefill only the *appended* tokens, so previous
    messages are encoded and stored exactly once instead of being re-prefilled (and re-stored)
    from scratch every turn.  Otherwise (first turn, ``/reset``, or a tokenization mismatch in the
    re-rendered reply) we start a fresh cache, which the router resets at ``start_pos == 0``.

    Returns ``(cache, prefill_start)``.
    """
    if (
        cache is not None
        and len(new_ids) > len(cached_ids)
        and new_ids[: len(cached_ids)] == cached_ids
    ):
        return cache, len(cached_ids)
    return DynamicCache(), 0


def _generate_iter(
    model, tok, input_ids: torch.Tensor, max_new: int, block: int,
    cache: Union[DynamicCache, None] = None, prefill_start: int = 0,
) -> Generator[Union[str, dict], None, None]:
    """Blocked prefill + greedy decode over a (optionally persistent) KV cache.

    ``cache`` is the per-session ``DynamicCache`` carrying the router/KV store; only
    ``input_ids[:, prefill_start:]`` is prefilled (the ``[:prefill_start]`` prefix is already
    resident).  Yields text deltas (str) as they are produced, then a final stats dict:
    {"n_ctx", "n_out", "ttft", "tok_s", "peak_gb", "cached_ids"}.  ``cached_ids`` is the full token
    sequence now resident in the cache (prompt + generated), for the caller to carry to next turn.
    """
    with torch.inference_mode():
        device = input_ids.device
        eos_ids = {tok.eos_token_id} if isinstance(tok.eos_token_id, int) else set()
        if cache is None:
            cache = DynamicCache()
            prefill_start = 0
        S = input_ids.shape[1]

        t0 = time.perf_counter()
        last = None
        for s in range(prefill_start, S, block):
            e = min(s + block, S)
            cp = torch.arange(s, e, device=device)
            out = model(
                input_ids=input_ids[:, s:e],
                past_key_values=cache,
                cache_position=cp,
                position_ids=cp.unsqueeze(0),
                use_cache=True,
            )
            last = out.logits[:, -1]
        torch.cuda.synchronize()
        ttft = time.perf_counter() - t0

        gen_ids: list[int] = []
        decoded = ""
        p = S
        nxt = int(last.argmax(-1))
        n_out = 0
        t_gen = time.perf_counter()
        for _ in range(max_new):
            if nxt in eos_ids:
                break
            gen_ids.append(nxt)
            text = tok.decode(gen_ids, skip_special_tokens=True)
            delta = text[len(decoded):]
            decoded = text
            if delta:
                yield delta
            cp = torch.tensor([p], device=device)
            out = model(
                input_ids=torch.tensor([[nxt]], device=device),
                past_key_values=cache,
                cache_position=cp,
                position_ids=cp.unsqueeze(0),
                use_cache=True,
            )
            nxt = int(out.logits[:, -1].argmax(-1))
            p += 1
            n_out += 1

        dt = time.perf_counter() - t_gen
        tok_s = (n_out - 1) / dt if n_out > 1 else 0.0
        yield {
            "n_ctx": S,
            "n_out": n_out,
            "ttft": ttft,
            "tok_s": tok_s,
            "peak_gb": gb(torch.cuda.max_memory_allocated()),
            "cached_ids": input_ids[0].tolist() + gen_ids,
        }


def stream_generate(model, tok, input_ids: torch.Tensor, max_new: int, block: int,
                    cache: Union[DynamicCache, None] = None, prefill_start: int = 0):
    """Terminal mode wrapper: prints deltas, returns (reply, n_tokens, ttft, cached_ids)."""
    full: list[str] = []
    stats: dict = {}
    for item in _generate_iter(model, tok, input_ids, max_new, block, cache, prefill_start):
        if isinstance(item, dict):
            stats = item
        else:
            print(item, end="", flush=True)
            full.append(item)
    return (
        "".join(full),
        stats.get("n_out", 0),
        stats.get("ttft", 0.0),
        stats.get("cached_ids", input_ids[0].tolist()),
    )


# ---------------------------------------------------------------------------
# Browser UI
# ---------------------------------------------------------------------------

class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _make_handler(req_q: "queue.Queue[tuple]"):
    html_bytes = _CHAT_HTML.encode()

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence access log
            pass

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html_bytes)))
                self.end_headers()
                self.wfile.write(html_bytes)
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            if self.path == "/reset":
                req_q.put(("reset", None, None, None))
                self.send_response(204)
                self.end_headers()
                return

            if self.path == "/chat":
                data = json.loads(body)
                msg = data.get("message", "")
                think = bool(data.get("thinking", False))
                reply_q: "queue.Queue" = queue.Queue()
                req_q.put(("chat", msg, think, reply_q))

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                try:
                    while True:
                        item = reply_q.get()
                        if item is None:
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                            break
                        self.wfile.write(
                            ("data: " + json.dumps(item) + "\n\n").encode()
                        )
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return

            self.send_error(404)

    return _Handler


def _run_ui(model, tok, host: str, port: int) -> None:
    """Start the HTTP server, then loop processing generation requests in the main thread."""
    import socket as _socket

    req_q: "queue.Queue[tuple]" = queue.Queue()
    server = _ThreadedHTTPServer((host, port), _make_handler(req_q))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    # Resolve the LAN IP for the "open on another machine" hint.
    try:
        lan_ip = _socket.gethostbyname(_socket.gethostname())
    except OSError:
        lan_ip = host if host != "0.0.0.0" else "localhost"

    local_url = f"http://127.0.0.1:{port}"
    lan_url   = f"http://{lan_ip}:{port}"
    print(f"UI ready:", flush=True)
    print(f"  local  : {local_url}", flush=True)
    print(f"  network: {lan_url}  (open this on other machines)", flush=True)
    print("Ctrl-C to quit", flush=True)
    webbrowser.open(local_url)

    history: list[dict] = []
    cache: Union[DynamicCache, None] = None
    cached_ids: list[int] = []

    while True:
        kind, msg, think, reply_q = req_q.get()

        if kind == "reset":
            history.clear()
            cache = None
            cached_ids = []
            continue

        # kind == "chat" — run generation in the main thread (CUDA context stays here)
        history.append({"role": "user", "content": msg})
        text = tok.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=think,
        )
        input_ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        new_ids = input_ids[0].tolist()
        cache, prefill_start = _plan_prefill(cache, cached_ids, new_ids)

        full: list[str] = []
        try:
            for item in _generate_iter(
                model, tok, input_ids, MAX_NEW_TOKENS, PREFILL_BLOCK, cache, prefill_start
            ):
                if isinstance(item, dict):
                    s = item
                    cached_ids = s["cached_ids"]
                    label = (
                        f"{s['n_ctx']} ctx | {s['n_out']} tokens | "
                        f"TTFT {s['ttft']:.1f}s | {s['tok_s']:.1f} tok/s | "
                        f"peak {s['peak_gb']:.1f}GB"
                    )
                    reply_q.put({"stats": label})
                else:
                    full.append(item)
                    reply_q.put({"text": item})
        except Exception as exc:
            reply_q.put({"error": str(exc)})
        finally:
            history.append({"role": "assistant", "content": "".join(full)})
            reply_q.put(None)  # sentinel → browser sees [DONE]


# ---------------------------------------------------------------------------
# Terminal chat
# ---------------------------------------------------------------------------

def _terminal_chat(model, tok) -> None:
    history: list[dict] = []
    thinking = False
    cache: Union[DynamicCache, None] = None
    cached_ids: list[int] = []

    print("Commands: /reset  /think  /exit")
    print("─" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("Bye.")
            break
        if user_input == "/reset":
            history.clear()
            cache = None
            cached_ids = []
            print("[history cleared]")
            continue
        if user_input == "/think":
            thinking = not thinking
            print(f"[thinking mode: {'ON' if thinking else 'OFF'}]")
            continue

        history.append({"role": "user", "content": user_input})
        text = tok.apply_chat_template(
            history, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        input_ids = tok(text, return_tensors="pt").input_ids.to("cuda")
        n_prompt = input_ids.shape[1]
        new_ids = input_ids[0].tolist()
        cache, prefill_start = _plan_prefill(cache, cached_ids, new_ids)

        print("\nAssistant: ", end="", flush=True)
        t_start = time.perf_counter()
        reply, n_out, ttft, cached_ids = stream_generate(
            model, tok, input_ids, MAX_NEW_TOKENS, PREFILL_BLOCK, cache, prefill_start
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t_start

        sustained = (n_out - 1) / (dt - ttft) if n_out > 1 and dt > ttft else 0.0
        print(
            f"\n[{n_prompt} ctx | {n_out} tokens | TTFT {ttft:.1f}s | "
            f"{sustained:.1f} tok/s | peak {gb(torch.cuda.max_memory_allocated()):.1f}GB]",
            flush=True,
        )
        history.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with Qwen3-30B (RAM-cached KvRouter)")
    ap.add_argument("--ui", action="store_true", help="Open browser UI instead of terminal chat")
    ap.add_argument("--host", default="0.0.0.0", help="UI server host (default: 0.0.0.0 = all interfaces)")
    ap.add_argument("--port", type=int, default=7860, help="UI server port (default: 7860)")
    ap.add_argument("--cache", choices=("ram", "fs", "vram"), default=CACHE_LOCATION,
                    help="Cold-KV tier: ram (host RAM), fs (RAM-bounded + NVMe spillover), vram")
    ap.add_argument("--ram-budget-gb", type=float, default=RAM_BUDGET_GB,
                    help="Host-RAM ceiling for the fs tier before chunks spill to disk")
    ap.add_argument("--fs-cache-dir", default=FS_CACHE_DIR,
                    help="Directory for fs-tier spill files (must be a real disk, not tmpfs)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA not available"

    print(f"Loading {MODEL} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype="auto",
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    n = replace_qwen_attention_with_router(
        model, cache_location=args.cache,
        keep_first=KEEP_FIRST, keep_last=KEEP_LAST, topk_chunks=TOPK_CHUNKS,
        topk_groups=TOPK_GROUPS, chunk_size=CHUNK_SIZE, group_size=GROUP_SIZE,
        vram_cache_chunks=VRAM_CACHE_CHUNKS, vram_summary_chunks=VRAM_SUMMARY_CHUNKS,
        vram_cache_reserve_gb=VRAM_CACHE_RESERVE_GB,
        ram_budget_gb=args.ram_budget_gb, fs_cache_dir=args.fs_cache_dir,
        dca_chunk=DCA_CHUNK, dca_local=DCA_LOCAL,
    )
    torch.cuda.synchronize()
    print(
        f"Loaded in {time.perf_counter() - t0:.1f}s  "
        f"({gb(torch.cuda.memory_allocated()):.1f}GB / "
        f"{gb(torch.cuda.get_device_properties(0).total_memory):.1f}GB VRAM)",
        flush=True,
    )
    _tier = {
        "ram": "KV cache lives in host RAM.",
        "fs": f"KV cache: {args.ram_budget_gb:g}GB host-RAM page cache + NVMe/disk spillover.",
        "vram": "KV cache lives in VRAM.",
    }.get(args.cache, "")
    print(
        f"Router on {n} layers: keep_first={KEEP_FIRST} ({KEEP_FIRST*CHUNK_SIZE} tok), "
        f"keep_last={KEEP_LAST} ({KEEP_LAST*CHUNK_SIZE} tok), topk_chunks={TOPK_CHUNKS} "
        f"({TOPK_CHUNKS*CHUNK_SIZE} tok); " + _tier
        + (f"  DCA on: chunk={DCA_CHUNK} tok, ceil={DCA_CHUNK + (DCA_LOCAL or DCA_CHUNK // 5)} tok."
           if DCA_CHUNK > 0 else "")
        + "\n",
        flush=True,
    )

    if args.ui:
        _run_ui(model, tok, args.host, args.port)
    else:
        _terminal_chat(model, tok)


if __name__ == "__main__":
    main()
