#!/usr/bin/env bash
#
# show_dataset_structure.sh
# Prints the directory structure under your dataset root.
#
# Usage:
#   ./show_dataset_structure.sh /path/to/dataset_root
#

set -euo pipefail

ROOT="${1:-}"
if [[ -z "$ROOT" || ! -d "$ROOT" ]]; then
  echo "Usage: $0 /path/to/dataset_root"
  exit 1
fi

echo "=== Original sequences ==="
find "$ROOT/original_sequences" -mindepth 1 -maxdepth 1 \
     -print | sed "s|^$ROOT/||" | sort

echo
echo "=== Manipulated sequences ==="
find "$ROOT/manipulated_sequences" -mindepth 1 -maxdepth 1 \
     -print | sed "s|^$ROOT/||" | sort

echo
echo "=== Video counts by category ==="
for dir in original_sequences manipulated_sequences; do
  echo "  $dir: $(find "$ROOT/$dir" -type f -name '*.mp4' | wc -l) videos"
done
