# Release validation

Validation date: 2026-07-25.

## Completed gates

- All public Python modules compile and the tracked-only package installs.
- The final model has 791 state tensors and 4,360,516 parameters with stable key
  order, tensor shapes, and seeded initial values.
- The Holo20k v9 epoch 4 checkpoint loads into the modular model with
  `strict=True`, zero missing keys, and zero unexpected keys.
- The concise paper YAML and the complete argument snapshot resolve to
  identical model, data, loss, and optimizer contracts.
- The complete local test suite covers CLI/config parsing, data contracts,
  checkpoint registries, dataset construction, model parameter
  identity, objective behavior, and Lightning epoch state.

## Real-batch GPU equivalence

The audit used two real Holo20k training structures, their tensor/graph/surface/
static caches, the released E4 state, and one NVIDIA RTX 4080. The final package
and a read-only external copy of the research implementation were loaded
independently.

| Stage | Compared tensors | Worst absolute difference |
|---|---:|---:|
| Evaluation forward | 107 | 0 |
| Training forward | 104 | 0 |
| Loss components | 14 | 0 |
| Parameter gradients | 634 | 5.96e-8 |
| Parameters after one AdamW step | 790 | 2.97e-6 |

The original modularization report is
[`docs/validation/e4_real_batch_modular_equivalence.json`](validation/e4_real_batch_modular_equivalence.json).
The runtime/factory refactor was independently rerun on the same two-case
RTX 4080 audit and is recorded in
[`docs/validation/e4_runtime_refactor_equivalence.json`](validation/e4_runtime_refactor_equivalence.json).
The final post-migration report is
[`docs/validation/final_external_oracle_equivalence.json`](validation/final_external_oracle_equivalence.json).
The temporary comparison script and external source tree are intentionally not
shipped with the release.

## Evaluation entry-point gate

The final `python main.py eval` command was run in the base CUDA environment
without Lightning installed. It strictly loaded E4, evaluated two real
cache-backed Holo20k cases, covered 611 valid residues with zero skipped cases,
and wrote metrics JSON/CSV, per-case CSV, and Gaussian diagnostic records. The
record is [`docs/validation/final_eval_smoke.json`](validation/final_eval_smoke.json).

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
1.79 seconds and stopped normally at `max_steps=1`. The machine-readable record
is [`docs/validation/lightning_real_cache_fast_dev.json`](validation/lightning_real_cache_fast_dev.json).

## Single-source policy

The release contains one implementation of each scientific operator. Model,
loss, data, preprocessing, evaluation, and Lightning training modules all live
under `src/m4`. CI rejects imports from removed research-tree module names.
