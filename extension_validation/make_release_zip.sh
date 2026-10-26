#!/bin/bash
# Repack the downloadable source/data bundle. Run ONLY after upstream A/B final
# results + HTML/README/final_tree are updated. Idempotent; overwrites the same
# 验证代码与数据.zip (explicitly allowed by owner).
set -euo pipefail
EV="/Users/bytedance/Desktop/Fermionic Gaussianity/extension_validation"
ROOT="/Users/bytedance/Desktop/Fermionic Gaussianity"
cd "$ROOT"

OUT="$EV/验证代码与数据.zip"
rm -f "$OUT"

# Include only: code + committed data + entrypoints + review notes.
# Exclude: report/ (delivered separately as HTML), .venv, runs/ (snapshot),
# _shots, __pycache__, *.pyc, all *.zip (no nesting), .git.
zip -r -q "$OUT" extension_validation \
  -x "extension_validation/.venv/*" \
  -x "extension_validation/runs/*" \
  -x "extension_validation/report/*" \
  -x "extension_validation/**/_shots/*" \
  -x "extension_validation/**/__pycache__/*" \
  -x "extension_validation/**/*.pyc" \
  -x "extension_validation/**/*.zip" \
  -x "extension_validation/.git/*"

echo "=== zip integrity test ==="
unzip -t "$OUT" | tail -2
echo "=== summary ==="
echo "path: $OUT"
ls -lh "$OUT" | awk '{print "size:",$5}'
unzip -l "$OUT" | tail -1
