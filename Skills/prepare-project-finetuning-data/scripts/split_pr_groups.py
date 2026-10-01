#!/usr/bin/env python3
"""Split PR-derived JSONL by whole PR, holding out the newest PR groups."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--train-out", type=Path, required=True)
    p.add_argument("--val-out", type=Path, required=True)
    p.add_argument("--n-val-prs", type=int, required=True)
    args = p.parse_args()
    if args.n_val_prs <= 0:
        p.error("--n-val-prs must be positive")
    paths = [x.resolve() for x in (args.input, args.train_out, args.val_out)]
    if len(set(paths)) != 3:
        p.error("input, train, and validation paths must differ")
    if any(path.exists() for path in paths[1:]):
        p.error("output already exists; choose new paths")

    groups: dict[str, list[dict]] = defaultdict(list)
    dates: dict[str, datetime] = {}
    for line_no, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("meta") or {}
        key = str(meta.get("pr_number", ""))
        raw_date = str(meta.get("date", ""))
        if not key or not raw_date:
            raise ValueError(f"line {line_no}: missing meta.pr_number or meta.date")
        date = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError(f"line {line_no}: date needs timezone")
        if key in dates and dates[key] != date:
            raise ValueError(f"line {line_no}: inconsistent date for PR {key}")
        dates[key] = date
        groups[key].append(row)
    if len(groups) <= args.n_val_prs:
        raise ValueError("need at least one training PR and the requested validation PRs")

    ordered = sorted(groups, key=lambda key: (dates[key], key))
    train_keys = ordered[:-args.n_val_prs]
    val_keys = ordered[-args.n_val_prs:]
    if max(dates[key] for key in train_keys) >= min(dates[key] for key in val_keys):
        raise ValueError("train and validation dates overlap; choose a different PR boundary")

    for path, keys in ((args.train_out, train_keys), (args.val_out, val_keys)):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for key in keys:
                for row in groups[key]:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "train_prs": len(train_keys), "train_rows": sum(len(groups[k]) for k in train_keys),
        "val_prs": len(val_keys), "val_rows": sum(len(groups[k]) for k in val_keys),
        "train_latest": max(dates[k] for k in train_keys).isoformat(),
        "val_earliest": min(dates[k] for k in val_keys).isoformat(),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
