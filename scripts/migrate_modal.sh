#!/usr/bin/env bash
# Migrate throng-runs Modal volume: download from old account or upload to new.
#
# Usage:
#   ./scripts/migrate_modal.sh download ~/throng_backup
#   ./scripts/migrate_modal.sh upload   ~/throng_backup
#
# Before download: modal token new  (old account)
# Before upload:   modal token new  (new account)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VOLUME="throng-runs"

resolve_modal() {
  if command -v modal >/dev/null 2>&1; then
    echo modal
    return
  fi
  local bundled="$ROOT/.modal-cli/bin/modal"
  if [[ -x "$bundled" ]]; then
    echo "$bundled"
    return
  fi
  echo "modal CLI not found. Install with: pip install modal" >&2
  exit 1
}

MODAL="$(resolve_modal)"

usage() {
  cat <<EOF
Usage:
  $0 download <target_dir>   Pull checkpoints + corpora from Modal volume
  $0 upload   <source_dir>   Create volume (if needed) and push local backup

Examples:
  $0 download ~/throng_backup
  $0 upload   ~/throng_backup
EOF
  exit 1
}

[[ $# -ge 2 ]] || usage

MODE="$1"
DIR="$2"
DIR="${DIR/#\~/$HOME}"

case "$MODE" in
  download)
    mkdir -p "$DIR"
    echo "=== Downloading $VOLUME → $DIR ==="
    "$MODAL" volume get --force "$VOLUME" checkpoints "$DIR/"
    "$MODAL" volume get --force "$VOLUME" signal_corpus.jsonl "$DIR/"
    "$MODAL" volume get --force "$VOLUME" signal_corpus_red.jsonl "$DIR/"
    echo "=== Download complete ==="
    ls -la "$DIR"
    ;;
  upload)
    for path in "$DIR/checkpoints" "$DIR/signal_corpus.jsonl" "$DIR/signal_corpus_red.jsonl"; do
      if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 1
      fi
    done
    echo "=== Ensuring volume $VOLUME exists ==="
    "$MODAL" volume create "$VOLUME" || true
    echo "=== Uploading $DIR → $VOLUME ==="
    "$MODAL" volume put "$VOLUME" "$DIR/checkpoints" /checkpoints
    "$MODAL" volume put "$VOLUME" "$DIR/signal_corpus.jsonl" /
    "$MODAL" volume put "$VOLUME" "$DIR/signal_corpus_red.jsonl" /
    echo "=== Upload complete ==="
    "$MODAL" volume ls "$VOLUME" /
    ;;
  *)
    usage
    ;;
esac
