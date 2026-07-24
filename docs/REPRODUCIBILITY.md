# Reproducibility

Reproducibility has four independent layers. Passing a later layer does not
repair a failure in an earlier one.

## 1. Immutable inputs

Record the Git commit, checkpoint SHA-256, manifest SHA-256, Pinder release,
test manifest hashes, four cache versions, and MMseqs2 command/report. The exact
Holo20k v9 manifest is an external release asset registered in
`assets/datasets.json`; a newly sampled dataset is not the same experiment even
if its aggregate statistics match.

## 2. Package and checkpoint integrity

```bash
python -m pytest -q
python scripts/validate_checkpoint.py CHECKPOINT \
  --registry-name m4_holo20k_v9_clean_epoch4
```

Strict checkpoint loading requires zero missing and unexpected keys. The
registry check additionally verifies bytes, SHA-256, parameter count, and state
tensor count.

## 3. Inference equivalence

The final release was compared against an external read-only research snapshot
on the same CUDA GPU and real cached mini-set. Evaluation and training forward
outputs were exactly equal. The machine-readable record is
`docs/validation/final_external_oracle_equivalence.json`; the external source
tree is deliberately not a runtime or release dependency.

## 4. Training equivalence

The same external audit compared all loss components, gradients, and one AdamW
update from identical initialization and batch. For reproduction, run the full
configured experiment with seed 42 and the three registered evaluation
tracks.

GPU kernels and mixed precision can differ across hardware. Cross-machine
claims must state GPU model, CUDA, PyTorch, precision, deterministic settings,
seed, and numerical tolerance. Future scientific changes must repeat the CUDA
forward/loss/gradient/update audit before a new release claims equivalence.
