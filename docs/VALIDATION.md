# Release validation

Validation date: 2026-07-25.

## Completed gates

- The frozen oracle files match all recorded SHA-256 digests.
- All public Python modules compile and the tracked-only package installs.
- The modular constructor and frozen oracle create the same 791 state tensors,
  4,360,516 parameters, key order, tensor shapes, and seeded initial values.
- The Holo20k v9 epoch 4 checkpoint loads into the modular model with
  `strict=True`, zero missing keys, and zero unexpected keys.
- The concise paper YAML and the 264-line frozen argument snapshot resolve to
  identical model, data, loss, and optimizer contracts.
- The complete local test suite covers CLI/config parsing, data contracts,
  source hashes, checkpoint registries, dataset construction, modular parameter
  identity, objective behavior, and Lightning epoch state.

## Real-batch GPU equivalence

The audit used two real Holo20k training structures, their tensor/graph/surface/
static caches, the released E4 state, and one NVIDIA RTX 4080. The modular model
and frozen oracle were loaded independently.

| Stage | Compared tensors | Worst absolute difference |
|---|---:|---:|
| Evaluation forward | 107 | 0 |
| Training forward | 104 | 0 |
| Loss components | 14 | 0 |
| Parameter gradients | 634 | 5.96e-8 |
| Parameters after one AdamW step | 790 | 1.07e-6 |

The full report is
[`docs/validation/e4_real_batch_modular_equivalence.json`](validation/e4_real_batch_modular_equivalence.json).
The reusable command is `scripts/validate_modular_equivalence.py`; it fails on
state-key drift, skipped cache cases, output/loss differences, gradient drift,
or optimizer-step drift.

## Lightning training gate

A separate `venv --system-site-packages` on the same server ran the public
`python main.py train` path with:

- Python 3.10.8;
- PyTorch 2.1.2+cu121 and CUDA 12.1;
- Lightning 2.6.5;
- mixed precision;
- batch size 2 and gradient accumulation 4; and
- the real Holo20k manifest plus all four cache families.

Lightning completed one forward/loss/backward/AdamW/scheduler step in about
2.27 seconds and stopped normally at `max_steps=1`. The machine-readable record
is [`docs/validation/lightning_real_cache_fast_dev.json`](validation/lightning_real_cache_fast_dev.json).

## Frozen oracle policy

`src/m4/legacy_repo` remains in the release only as a SHA-pinned oracle and as
the dependency closure for low-level historical operators/cache formats. The
paper-facing model and training paths are now `src/m4/models`, `src/m4/losses.py`,
`src/m4/data`, and `src/m4/lightning`. New scientific logic must not be added to
the legacy trainer.
