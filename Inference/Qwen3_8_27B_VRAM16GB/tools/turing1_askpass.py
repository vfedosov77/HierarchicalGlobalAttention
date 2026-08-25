#!/usr/bin/env python3
"""SSH_ASKPASS helper: print the turing1 password from secrets.md.

OpenSSH invokes this with a prompt on argv; stdout must be the password only.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SECRETS = Path(__file__).resolve().parents[1] / "secrets.md"
PASSWORD_RE = re.compile(r"(?im)^\s*#\s*turing1 password:\s*(\S+)")


def main() -> int:
    if not SECRETS.is_file():
        sys.stderr.write(f"missing {SECRETS}\n")
        return 1
    match = PASSWORD_RE.search(SECRETS.read_text(encoding="utf-8"))
    if not match:
        sys.stderr.write(f"{SECRETS} has no '# turing1 password:' line\n")
        return 1
    sys.stdout.write(match.group(1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
