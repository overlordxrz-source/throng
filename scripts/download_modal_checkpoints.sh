#!/usr/bin/env bash
# Download throng-runs/checkpoints from Modal to ~/throng_checkpoints_backup
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODAL="$ROOT/.modal-cli/bin/modal"
OUT="${1:-$HOME/throng_checkpoints_backup}"

if [[ ! -x "$MODAL" ]]; then
  echo "Installing Modal CLI..."
  python3 -m venv "$ROOT/.modal-cli"
  "$ROOT/.modal-cli/bin/pip" install -q modal
fi

echo "=== Modal auth (opens browser — use your OLD account) ==="
"$MODAL" token new

echo ""
echo "=== Volumes ==="
"$MODAL" volume list

echo ""
echo "=== Downloading throng-runs/checkpoints → $OUT ==="
mkdir -p "$OUT"
"$MODAL" volume get throng-runs checkpoints "$OUT"

echo ""
echo "=== Done. Latest checkpoint folders: ==="
find "$OUT" -maxdepth 3 -type d -name '[0-9]*' 2>/dev/null | sort -V | tail -5 || ls -la "$OUT"
