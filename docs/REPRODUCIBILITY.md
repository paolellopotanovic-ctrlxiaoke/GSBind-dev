# Reproducibility

Reproducibility has four independent layers. Passing a later layer does not
repair a failure in an earlier one.

## 1. Immutable inputs

Record the Git commit, checkpoint SHA-256, manifest SHA-256, Pinder release,
test manifest hashes, four cache versions, and MMseqs2 command/report. The exact
Holo20k v9 manifest is an external release asset registered in
`assets/datasets.json`; a newly sampled dataset is not the same experiment even
if its aggregate statistics match.

## 2. Source and checkpoint integrity

```bash
python scripts/verify_sources.py
python scripts/validate_checkpoint.py CHECKPOINT \
  --registry-name m4_holo20k_v9_clean_epoch4
```

Strict checkpoint loading requires zero missing and unexpected keys. The
registry check additionally verifies bytes, SHA-256, parameter count, and state
tensor count.

## 3. Inference equivalence

On the same CUDA GPU and cached mini-set, run
`scripts/validate_server_equivalence.sh`. The labeled evaluator from the
research tree and release package must produce the same load report, metrics,
case metrics, and diagnostic records. Also compare raw logits and record maximum
absolute and relative differences; exact equality is preferred in the pinned
environment.

## 4. Training equivalence

Run one deterministic optimizer step from the same initialization and batch.
Record input cache hashes, RNG state, CUDA/PyTorch versions, AMP scaler state,
loss components, gradients, optimizer state, and updated weights. Then run the
full configured experiment with seed 42 and the three registered evaluation
tracks.

GPU kernels and mixed precision can differ across hardware. Cross-machine
claims must state GPU model, CUDA, PyTorch, precision, deterministic settings,
seed, and numerical tolerance. Until layers 3 and 4 pass on CUDA, this repository
must not claim complete numerical equivalence.
