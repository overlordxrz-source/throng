#!/bin/zsh
# stamp.sh — the ONLY way protocol / handoff times are written (Cam Nov 16 ~12:30Z, item 4).
#
# Usage:  scripts/stamp.sh FILE ANCHOR_REGEX [--log-times] < entry.md
#   The entry comes on stdin. Every @NOW@ in it is replaced by the output of `date` (taken here, at insertion).
#   ANCHOR_REGEX: the entry is inserted immediately BEFORE the single line matching it (Python regex; it must match
#   exactly one line, or nothing is written). ANCHOR_REGEX = END appends at the end of the file.
#   A hand-typed clock time (hh:mm) in the entry is refused. --log-times allows hh:mm values that are COPIED from a
#   named log or file (say which in the entry, e.g. "06:18:01 from queue_rl.out").
# Prints the stamped time and the line number of the inserted entry.
set -e
F=$1; A=$2; LOGT=${3:-}
[ -f "$F" ] || { echo "stamp: no file $F" >&2; exit 2; }
[ -n "$A" ] || { echo "stamp: need an anchor regex (or END)" >&2; exit 2; }
NOW=$(date '+%b %d %H:%M:%S %Z')
TMP=$(mktemp)
cat > "$TMP"
python3 - "$F" "$A" "$NOW" "$TMP" "$LOGT" <<'PY'
import re, sys
f, anchor, now, tmp, logt = sys.argv[1:6]
entry = open(tmp).read()
if "@NOW@" not in entry:
    sys.exit("stamp: the entry has no @NOW@ placeholder; nothing written")
if logt != "--log-times" and re.search(r"\b\d{1,2}:\d{2}\b", entry):
    sys.exit("stamp: the entry contains a typed clock time (hh:mm); use @NOW@, or --log-times for times copied from a log")
entry = entry.replace("@NOW@", now)
if not entry.endswith("\n"):
    entry += "\n"
lines = open(f).read().splitlines(keepends=True)
if anchor == "END":
    at = len(lines)
else:
    hits = [i for i, l in enumerate(lines) if re.search(anchor, l)]
    if len(hits) != 1:
        sys.exit(f"stamp: anchor matched {len(hits)} lines (need exactly 1); nothing written")
    at = hits[0]
lines[at:at] = [entry]
open(f, "w").write("".join(lines))
print(f"stamped {now} -> {f}:{at + 1}")
PY
rm -f "$TMP"
