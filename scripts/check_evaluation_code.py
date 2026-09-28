"""Has the evaluation (or training) code changed since a baseline, beyond reviewed and approved changes? (read-only)

Run from the repo root with the RL interpreter:
    .venv-rl/Scripts/python.exe scripts/check_evaluation_code.py --baseline f26914d --approved 44ffab5
    .venv-rl/Scripts/python.exe scripts/check_evaluation_code.py --entry scripts/train_room1.py --baseline 331e6c9 \
        --approved 44ffab5          # the training code a new arm shares with an earlier control

With --entry, the checked code is those scripts and their import closure instead of the evaluators'.

Limit: several --approved commits can only be combined when they change different files; a second approved change
to the same file needs a new baseline (their concatenated diffs would not equal the single diff over both).

The evaluation code is every project file that scripts/evaluate_heldout.py and scripts/evaluate_checkpoint.py load
(their import closure, found by importing them), plus the two scripts. The check passes when the diff of those files
from --baseline to HEAD is byte-for-byte the diff that the --approved commits made to them (compared by sha256), so
any other change to the evaluation code, however small, fails it. With no --approved, the files must be unchanged.
Also refuses uncommitted changes in those files, since HEAD would not describe them.

Why: results are compared with floors and before-values measured by earlier evaluations. A reviewed change proven not
to alter results (the side-by-side port defaults of 44ffab5, identical on 1,600 + 6,808 evaluation episodes) may be
allowed; nothing else passes without its own review, proof, or a re-measure.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVALUATORS = ["scripts/evaluate_heldout.py", "scripts/evaluate_checkpoint.py"]


def closure(entries: list[str] = EVALUATORS) -> list[str]:
    """Repo-relative paths of the entry scripts and every celeste_rl module they import (found by importing them)."""
    modules = ", ".join(Path(entry).stem for entry in entries)
    code = (f"import os, sys; sys.path.insert(0, 'scripts'); import {modules}; "
            "print('\\n'.join(sorted({os.path.relpath(m.__file__).replace(os.sep, '/') for n, m in sys.modules.items() "
            "if n.startswith('celeste_rl') and getattr(m, '__file__', None)})))")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True, env=env)
    return sorted(set(out.stdout.split()) | set(entries))


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, check=True).stdout


def diff_sha256(files: list[str], *revisions: str) -> str:
    return hashlib.sha256(git("diff", *revisions, "--", *files)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="the commit the compared evaluations ran at")
    parser.add_argument("--approved", nargs="*", default=[], help="reviewed commits whose changes are allowed")
    parser.add_argument("--entry", nargs="+", default=EVALUATORS,
                        help="entry scripts whose code is checked (default: the two evaluators)")
    args = parser.parse_args()
    files = closure(args.entry)
    dirty = git("status", "--porcelain", "--", *files).decode().strip()
    if dirty:
        print(f"REFUSED: uncommitted changes in the evaluation code:\n{dirty}")
        return 2
    actual = diff_sha256(files, args.baseline, "HEAD")
    allowed = hashlib.sha256(b"".join(git("diff", f"{c}^", c, "--", *files) for c in args.approved)).hexdigest()
    changed = git("diff", "--name-only", args.baseline, "HEAD", "--", *files).decode().split()
    print(f"code of {', '.join(args.entry)}: {len(files)} files; changed since {args.baseline}: {changed or 'none'}")
    print(f"diff since baseline {actual[:16]}; approved changes {allowed[:16]} ({', '.join(args.approved) or 'none'})")
    if actual == allowed:
        print("PASS: the code equals the baseline plus exactly the approved changes")
        return 0
    print("FAIL: the code differs from the baseline beyond the approved changes")
    return 1


if __name__ == "__main__":
    sys.exit(main())
