#!/usr/bin/env bash
set -euo pipefail

: "${M4_REFERENCE_ROOT:?set M4_REFERENCE_ROOT to the validated PeSTo-main tree}"
: "${M4_RELEASE_ROOT:?set M4_RELEASE_ROOT to this release checkout}"
: "${M4_CHECKPOINT:?set M4_CHECKPOINT}"
: "${M4_TRAIN_MANIFEST:?set M4_TRAIN_MANIFEST}"
: "${M4_EVAL_MANIFEST:?set M4_EVAL_MANIFEST}"
: "${M4_TENSOR_CACHE:?set M4_TENSOR_CACHE}"
: "${M4_GRAPH_CACHE:?set M4_GRAPH_CACHE}"
: "${M4_SURFACE_CACHE:?set M4_SURFACE_CACHE}"
: "${M4_STATIC_CACHE:?set M4_STATIC_CACHE}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${M4_DEVICE:-cpu}"
OUTPUT_ROOT="${M4_VALIDATION_OUTPUT:-$M4_RELEASE_ROOT/.validation/equivalence}"

common=(
  --manifest "$M4_EVAL_MANIFEST"
  --checkpoint "$M4_CHECKPOINT"
  --train-manifest "$M4_TRAIN_MANIFEST"
  --tensor-cache-dir "$M4_TENSOR_CACHE"
  --residue-graph-cache-dir "$M4_GRAPH_CACHE"
  --s1-surface-cache-dir "$M4_SURFACE_CACHE"
  --s1-static-geometry-cache-dir "$M4_STATIC_CACHE"
  --require-s1-static-geometry-cache
  --require-s1-surface-cache
  --disable-s1-surface-on-the-fly
  --s1-surface-cache-version s1_surface_shrake_v1
  --states apo,af2
  --device "$DEVICE"
  --batch-size 1
  --num-workers 0
  --matmul-precision high
  --s1-surface-points 1024
  --s1-surface-graph-k 16
  --surface-extra-graph-ks 32
  --surface-global-feedback-before-residue
  --max-skip-fraction 0.01
  --max-cases 1
)

mkdir -p "$OUTPUT_ROOT/old" "$OUTPUT_ROOT/new"

cd "$M4_REFERENCE_ROOT"
PYTHONPATH="$M4_REFERENCE_ROOT:$M4_REFERENCE_ROOT/model" "$PYTHON_BIN" -u \
  model/apex_s1_surface_field/eval_s1_surface_field.py "${common[@]}" \
  --output-dir "$OUTPUT_ROOT/old" --prefix equivalence \
  > "$OUTPUT_ROOT/old.log" 2>&1

cd "$M4_RELEASE_ROOT"
"$PYTHON_BIN" -u main.py eval "${common[@]}" \
  --output-dir "$OUTPUT_ROOT/new" --prefix equivalence \
  > "$OUTPUT_ROOT/new.log" 2>&1

"$PYTHON_BIN" - "$OUTPUT_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
names = (
    "load_state_report.json",
    "equivalence_metrics.json",
    "equivalence_metrics.csv",
    "equivalence_case_metrics.csv",
    "equivalence_gaussian_top_records.jsonl",
)
different = []
for name in names:
    old = (root / "old" / name).read_bytes()
    new = (root / "new" / name).read_bytes()
    status = "IDENTICAL" if old == new else "DIFF"
    print(status, name, hashlib.sha256(old).hexdigest(), hashlib.sha256(new).hexdigest())
    if old != new:
        different.append(name)
    if name.endswith(".json"):
        print(json.loads(old))
if different:
    raise SystemExit(f"non-equivalent outputs: {different}")
PY
