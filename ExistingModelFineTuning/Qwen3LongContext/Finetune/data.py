"""Data sources for long-context HGA fine-tuning: a long-context corpus, a dialog corpus, and
the user-provided novel (The Master and Margarita), packed into fixed-length token sequences and
mixed by weight.

Each *source* is an infinite generator of 1-D ``LongTensor`` of exactly ``seq_len`` token ids
(model tokenizer).  :class:`MixedBatcher` samples ``batch_size`` sequences per step by weight and
stacks them to ``[B, seq_len]``.  Per-sequence independence (no cross-sequence chunk mixing) is
guaranteed downstream by the store's batch dimension — every sequence is its own batch row.

Sources
-------
* ``mm``     — The-Master-and-Margarita.txt, random ``seq_len`` windows (offline, always available).
* ``long``   — local fineweb sample parquet, docs with ``token_count >= seq_len`` (offline).
* ``dialog`` — a streaming HF chat/Q-A dataset (default ``HuggingFaceH4/ultrachat_200k``), rendered
               with the model chat template and packed to ``seq_len`` (needs network on first run;
               degrades gracefully to ``None`` if unavailable).
"""
from __future__ import annotations

import glob
import os
import random
from typing import Iterator, List, Optional

import torch

Seq = torch.Tensor  # 1-D LongTensor [seq_len]


# =================================================================================================
# Individual sources (each an infinite generator of [seq_len] LongTensors)
# =================================================================================================
def mm_source(txt_path: str, tokenizer, seq_len: int, seed: int = 0) -> Iterator[Seq]:
    """Random ``seq_len`` windows over the tokenized novel (cycles forever)."""
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    n = int(ids.numel())
    if n <= seq_len:  # tile so we can always cut a full window
        ids = ids.repeat(seq_len // max(n, 1) + 2)
        n = int(ids.numel())
    rng = random.Random(seed)
    while True:
        start = rng.randint(0, n - seq_len - 1)
        yield ids[start : start + seq_len].clone()


def long_source(parquet_glob: str, tokenizer, seq_len: int, seed: int = 0,
                min_tokens: Optional[int] = None) -> Iterator[Seq]:
    """Random ``seq_len`` windows from fineweb docs long enough to fill a full sequence."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(parquet_glob))
    if not files:
        raise FileNotFoundError(f"no parquet files match {parquet_glob}")
    min_tokens = min_tokens if min_tokens is not None else seq_len
    rng = random.Random(seed)
    while True:
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=64, columns=["text", "token_count"]):
                d = batch.to_pydict()
                for text, tc in zip(d["text"], d["token_count"]):
                    if tc is None or tc < min_tokens:
                        continue
                    ids = tokenizer(text, return_tensors="pt",
                                    add_special_tokens=False).input_ids[0]
                    if ids.numel() < seq_len:
                        continue
                    start = rng.randint(0, int(ids.numel()) - seq_len)
                    yield ids[start : start + seq_len].clone()


def dialog_source(tokenizer, seq_len: int, *, name: str = "HuggingFaceH4/ultrachat_200k",
                  config: Optional[str] = None, split: str = "train_sft",
                  seed: int = 0) -> Optional[Iterator[Seq]]:
    """Stream a chat/Q-A dataset, render with the chat template, pack to ``seq_len``.

    Returns ``None`` (with a warning) if the dataset cannot be loaded, so training can proceed
    with the offline sources only.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(name, config, split=split, streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=1000)
    except Exception as e:  # pragma: no cover - network/availability dependent
        print(f"[data] dialog source '{name}' unavailable ({e}); skipping it.")
        return None

    def _render(ex) -> List[int]:
        msgs = ex.get("messages") or ex.get("conversations") or ex.get("conversation")
        if msgs is None and "prompt" in ex and "response" in ex:
            msgs = [{"role": "user", "content": ex["prompt"]},
                    {"role": "assistant", "content": ex["response"]}]
        if not msgs:
            return []
        try:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        except Exception:
            text = "\n".join(str(m.get("content", "")) for m in msgs)
        return tokenizer(text, add_special_tokens=False).input_ids

    def _gen() -> Iterator[Seq]:
        buf: List[int] = []
        while True:
            for ex in ds:
                buf.extend(_render(ex))
                while len(buf) >= seq_len:
                    yield torch.tensor(buf[:seq_len], dtype=torch.long)
                    buf = buf[seq_len:]

    return _gen()


# =================================================================================================
# Mixer
# =================================================================================================
class MixedBatcher:
    """Samples ``batch_size`` sequences per step from weighted sources → ``[B, seq_len]``."""

    def __init__(self, sources: List[Iterator[Seq]], weights: List[float],
                 batch_size: int, seq_len: int, seed: int = 0) -> None:
        assert sources and len(sources) == len(weights)
        self.sources = sources
        self.weights = weights
        self.B = batch_size
        self.seq_len = seq_len
        self.rng = random.Random(seed)

    def next_batch(self) -> torch.Tensor:
        rows = []
        for _ in range(self.B):
            i = self.rng.choices(range(len(self.sources)), weights=self.weights, k=1)[0]
            rows.append(next(self.sources[i]))
        return torch.stack(rows, dim=0)  # [B, seq_len]


def build_sources(names: List[str], weights: List[float], *, tokenizer, seq_len: int,
                  mm_path: str, parquet_glob: str, dialog_name: str, dialog_split: str,
                  seed: int = 0):
    """Construct the requested sources, dropping any that are unavailable (with their weight)."""
    out_sources: List[Iterator[Seq]] = []
    out_weights: List[float] = []
    out_names: List[str] = []
    for name, w in zip(names, weights):
        if name == "mm":
            gen: Optional[Iterator[Seq]] = mm_source(mm_path, tokenizer, seq_len, seed=seed)
        elif name == "long":
            gen = long_source(parquet_glob, tokenizer, seq_len, seed=seed)
        elif name == "dialog":
            gen = dialog_source(tokenizer, seq_len, name=dialog_name, split=dialog_split, seed=seed)
        else:
            raise ValueError(f"unknown source '{name}'")
        if gen is None:
            continue
        out_sources.append(gen)
        out_weights.append(w)
        out_names.append(name)
    if not out_sources:
        raise RuntimeError("no usable data sources")
    return out_sources, out_weights, out_names
