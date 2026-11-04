"""Config audit: list every config.yaml key that no code reads.

Standing pre-launch test (Cam, 2026-11-04): five settings so far were configured but never applied
(hearth shaping, the epistemic gate, the channel, the material grids, action_entropy_coef). Run before
every launch:

    python scripts/config_audit.py            # exit 1 if any unread key is not in the allowlist

A key counts as READ only if its leaf name appears in an access pattern -- `.get("key"`, `["key"]`
(either quote style) -- in a .py file under jax_sim/ or scripts/. A bare string literal does not count
(default dicts, fingerprint label lists and log text name keys without reading them).

Limitation, stated so nobody over-trusts a pass: READ is not APPLIED. A key that is read into a
variable the code never uses passes this audit (that was reward_red_catch's failure mode). This catches
the cheaper class -- keys nothing even reads.

Keys that are deliberately unread (documentation, read under a computed name, consumed by a tool
outside these trees) go in scripts/config_audit_allowlist.txt, one dotted key per line with a reason:
    some_section.some_key   # read as f"reward_{name}" in jax_sim/foo.py
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE_DIRS = ("jax_sim", "scripts")
ALLOWLIST = ROOT / "scripts" / "config_audit_allowlist.txt"


def leaf_keys(d: dict, prefix: str = ""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from leaf_keys(v, f"{prefix}{k}.")
        else:
            yield f"{prefix}{k}"


def code_text() -> str:
    me = pathlib.Path(__file__).resolve()
    parts = []
    for d in CODE_DIRS:
        for p in sorted((ROOT / d).rglob("*.py")):
            if p.resolve() != me:
                parts.append(p.read_text(errors="ignore"))
    return "\n".join(parts)


def is_read(leaf: str, text: str) -> bool:
    q = re.escape(leaf)
    pat = rf"""\.get\(\s*["']{q}["']|\[\s*["']{q}["']\s*\]"""
    return re.search(pat, text) is not None


def load_allowlist() -> set[str]:
    if not ALLOWLIST.exists():
        return set()
    out = set()
    for line in ALLOWLIST.read_text().splitlines():
        key = line.split("#", 1)[0].strip()
        if key:
            out.add(key)
    return out


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    text = code_text()
    allow = load_allowlist()
    keys = list(leaf_keys(cfg))
    unread = [k for k in keys if not is_read(k.rsplit(".", 1)[-1], text)]
    flagged = [k for k in unread if k not in allow]
    print(f"config_audit: {len(keys)} keys, {len(unread)} unread, {len(unread) - len(flagged)} allowlisted")
    for k in flagged:
        print(f"  UNREAD  {k}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
