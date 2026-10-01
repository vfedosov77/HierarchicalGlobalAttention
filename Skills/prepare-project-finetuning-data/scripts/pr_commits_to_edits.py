#!/usr/bin/env python3
"""Export local merged-PR commit ranges as single-turn whole-file edit JSONL.

This reconstructs pre/post files, never agent tool calls or reasoning. Input manifest:
{"number": 1, "title": "...", "base_sha": "...", "head_sha": "...",
 "merged_at": "2026-01-01T00:00:00+00:00"}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def commit(repo: Path, value: str) -> str:
    if not value or value.startswith("-"):
        raise ValueError("empty or invalid commit name")
    out = git(repo, "rev-parse", "--verify", f"{value}^{{commit}}").stdout.decode().strip()
    if len(out) != 40:
        raise ValueError(f"unexpected commit ID: {value}")
    return out


def file_at(repo: Path, sha: str, path: str) -> bytes | None:
    result = git(repo, "show", f"{sha}:{path}", check=False)
    return result.stdout if result.returncode == 0 else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-file-bytes", type=int, default=131072)
    args = p.parse_args()

    if args.max_file_bytes <= 0:
        p.error("--max-file-bytes must be positive")
    repo = args.repo.resolve()
    if git(repo, "rev-parse", "--is-inside-work-tree").stdout.strip() != b"true":
        p.error("--repo must be a Git worktree")
    if args.output.resolve() in {args.manifest.resolve(), repo.resolve()}:
        p.error("output collides with input or repository")

    rows: list[dict] = []
    skipped: dict[str, int] = {}
    seen_pr: set[str] = set()

    for line_no, line in enumerate(args.manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        for key in ("number", "title", "base_sha", "head_sha", "merged_at"):
            if key not in item or item[key] in (None, ""):
                raise ValueError(f"manifest line {line_no}: missing {key}")
        pr_number = str(item["number"])
        if pr_number in seen_pr:
            raise ValueError(f"manifest line {line_no}: duplicate PR {pr_number}")
        seen_pr.add(pr_number)
        merged_at = str(item["merged_at"])
        parsed_date = datetime.fromisoformat(merged_at.replace("Z", "+00:00"))
        if parsed_date.tzinfo is None:
            raise ValueError(f"manifest line {line_no}: merged_at needs timezone")
        base = commit(repo, str(item["base_sha"]))
        head = commit(repo, str(item["head_sha"]))
        if git(repo, "merge-base", "--is-ancestor", base, head, check=False).returncode != 0:
            raise ValueError(f"manifest line {line_no}: base is not an ancestor of head")
        paths = git(repo, "diff", "--name-only", "--no-renames", "--diff-filter=AM", "-z", base, head, "--").stdout
        for raw_path in filter(None, paths.split(b"\0")):
            path = os.fsdecode(raw_path)
            before = file_at(repo, base, path)
            after = file_at(repo, head, path)
            if after is None:
                skipped["missing_after"] = skipped.get("missing_after", 0) + 1
                continue
            was_added = before is None
            if was_added:
                before = b""
            if max(len(before), len(after)) > args.max_file_bytes:
                skipped["too_large"] = skipped.get("too_large", 0) + 1
                continue
            if b"\0" in before or b"\0" in after:
                skipped["binary"] = skipped.get("binary", 0) + 1
                continue
            try:
                old_text = before.decode("utf-8")
                new_text = after.decode("utf-8")
            except UnicodeDecodeError:
                skipped["non_utf8"] = skipped.get("non_utf8", 0) + 1
                continue
            if old_text == new_text:
                skipped["unchanged"] = skipped.get("unchanged", 0) + 1
                continue
            task = "Create" if was_added else "Edit"
            prompt = (
                f"Task from merged PR #{pr_number}: {item['title']}\n"
                f"{task} the complete file {path}. Return the complete resulting file only.\n"
                f"Pre-change file:\n{old_text}"
            )
            rows.append({
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": new_text},
                ],
                "tools": [],
                "meta": {
                    "source": "merged_pr_whole_file_edit",
                    "pr_number": item["number"],
                    "date": merged_at,
                    "base_sha": base,
                    "head_sha": head,
                    "path": path,
                    "status": "A" if was_added else "M",
                },
            })

    if not rows:
        raise ValueError(f"no usable edits; skipped={skipped}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(args.output, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"prs": len(seen_pr), "records": len(rows), "skipped": skipped,
                      "output": str(args.output), "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, subprocess.CalledProcessError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
