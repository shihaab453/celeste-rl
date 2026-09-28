"""Line-ending-safe hashes for committed text files (plans, configs, scripts).

On this machine git checks text files out with Windows line endings (core.autocrlf, `* text=auto`), while a file
written by a script keeps Unix endings until its next checkout. A plain sha256 of the working-tree file therefore
changes when the file is checked out again, although its content has not, and a recorded hash can be either form
(the coverage-ceiling plan's is CRLF, the mixed self-distillation plans' are LF). So:

- `text_sha256` hashes the content with CRLF turned into LF: the same value whichever way the file is checked out.
  New records of text files use it.
- `matches_text_hash` accepts a recorded hash that equals the file's raw, LF or CRLF form, so every existing record
  keeps verifying after a checkout changes the endings.

Binary files (checkpoints, datasets) are never touched by line-ending conversion; hash them as bytes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def text_sha256(path: Path) -> str:
    return hashlib.sha256(_lf(Path(path).read_bytes())).hexdigest()


def text_hash_forms(path: Path) -> set[str]:
    """The file's sha256 as it is, with LF endings, and with CRLF endings."""
    raw = Path(path).read_bytes()
    lf = _lf(raw)
    return {hashlib.sha256(form).hexdigest() for form in (raw, lf, lf.replace(b"\n", b"\r\n"))}


def matches_text_hash(path: Path, recorded: str | None) -> bool:
    return recorded is not None and recorded in text_hash_forms(path)
