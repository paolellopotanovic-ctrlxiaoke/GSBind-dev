# Gaussian-PPBS (M4)

[![CI](https://github.com/paolellopotanovic-ctrlxiaoke/Guassion-PPBS/actions/workflows/ci.yml/badge.svg)](https://github.com/paolellopotanovic-ctrlxiaoke/Guassion-PPBS/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/paolellopotanovic-ctrlxiaoke/Guassion-PPBS?include_prereleases)](https://github.com/paolellopotanovic-ctrlxiaoke/Guassion-PPBS/releases)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)

Gaussian-PPBS is the public implementation of **M4**, a receptor-only model for
protein-protein binding-site prediction. Given one protein structure, M4 assigns
an interface probability to every residue. A binding partner and biological
labels are not required at inference time.

M4 combines residue geometry, multi-scale residue graphs, a 1,024-point
solvent-accessible surface representation, and a learned Gaussian surface field.
This repository preserves the exact model tensors and checkpoint keys used by
the validated research implementation while exposing a small command-line API
for dataset construction, caching, training, evaluation, and label-free
inference.

> **Scope.** M4 predicts protein-protein interface residues. It is a
> prioritization model, not a docking engine, affinity predictor, or causal
> mechanism detector.

## Repository contents

```text
main.py                     unified train/eval/cache/data-entry CLI
inference.py                label-free per-residue prediction CLI
src/m4/                     public API and exact validated M4 implementation
configs/                    paper training and fine-tuning configurations
scripts/run_1crn_example.py one-command checkpoint-to-forward example
scripts/dataset/            Pinder construction, union, selection, and audits
assets/                     immutable checkpoint and dataset registries
examples/inference/         included 1CRN receptor-only example
docs/                       architecture, data, experiment, and validation docs
tests/                      unit and publication-contract tests
```

## Requirements

- Linux or Windows with Python 3.10
- NVIDIA GPU for the static-geometry cache and practical M4 inference
- CUDA 12.1 and PyTorch 2.1.2+cu121 in the validated environment
- Approximately 1 GB of free disk for the example environment, checkpoint, and
  generated caches (the checkpoint itself is 46.4 MB)

The CPU-only path can inspect source hashes and strictly validate the
checkpoint, but the complete surface/static-cache/forward example requires
CUDA.

## Quick start: one real forward pass

### 1. Clone and install

```bash
git clone https://github.com/paolellopotanovic-ctrlxiaoke/Guassion-PPBS.git
cd Guassion-PPBS
conda env create -f environment.yml
conda activate m4
python scripts/verify_sources.py
```

### 2. Run the bundled 1CRN example

```bash
python scripts/run_1crn_example.py
```

This single command performs the complete inference chain:

1. downloads `m4_holo20k_v9_clean_epoch4.pt` from the GitHub Release;
2. verifies its byte size and SHA-256;
3. strictly loads the checkpoint with zero missing/unexpected keys;
4. converts `examples/inference/1CRN.pdb` into a label-free manifest;
5. builds tensor, residue-graph, SHRake surface, and static-geometry caches;
6. runs one M4 forward pass; and
7. writes one probability row per PDB residue.

Expected outputs:

```text
work/1CRN/
|-- manifest.csv
|-- tensor/
|-- graph/
|-- surface/
|-- static/
`-- predictions/
    |-- residue_predictions.csv
    `-- prediction_metadata.json
```

Inspect the highest-scoring residues:

```bash
python -c "import pandas as pd; p=pd.read_csv('work/1CRN/predictions/residue_predictions.csv'); print(p.nlargest(10, 'interface_probability').to_string(index=False))"
```

On a CPU-only machine, print and audit the exact command chain without running
GPU work:

```bash
python scripts/run_1crn_example.py --dry-run
```

Use a previously downloaded checkpoint or run a receptor-only PDB of your own:

```bash
python scripts/run_1crn_example.py --pdb path/to/receptor.pdb --case-id MY_PROTEIN --input-state apo --checkpoint /path/to/model.pt --work-dir work/MY_PROTEIN
```

## Checkpoint

The primary inference checkpoint is stored as a GitHub Release asset, never as
an ordinary Git blob:

```bash
python scripts/download_checkpoint.py m4_holo20k_v9_clean_epoch4
python scripts/validate_checkpoint.py checkpoints/m4_holo20k_v9_clean_epoch4.pt --registry-name m4_holo20k_v9_clean_epoch4
```

| Field | Value |
|---|---:|
| Epoch / global step | 4 / 10,000 |
| Parameters | 4,360,516 |
| State tensors | 791 |
| File size | 46,437,755 bytes |
| SHA-256 | `d6b418e21d8fc43f347f7c4438e2339877dd3afb240c6196aa7eff896caaee14` |

The downloader refuses a file with a different size or checksum. The validator
also requires strict loading with no missing or unexpected keys.

## Inference on your own structure

The input must be a receptor-only PDB file with standard protein atoms and
stable chain/residue numbering. Replace the example PDB and case ID:

```bash
python main.py prepare-input --pdb path/to/receptor.pdb --case-id MY_PROTEIN --input-state apo --output work/MY_PROTEIN/manifest.csv
```

Then build the same four cache families used during training and call
`inference.py`. The exact manual commands are documented in
[`examples/inference/README.md`](examples/inference/README.md). Inference fails
closed on missing caches, residue-order mismatches, skipped cases, or non-strict
checkpoint loading.

`residue_predictions.csv` contains identifiers and probabilities including:

- case, chain, PDB residue number, insertion code, and amino acid;
- validity mask;
- final interface probability;
- Gaussian prior probability; and
- surface-conditioned probability.

`prediction_metadata.json` records the checkpoint SHA-256, epoch, device,
number of cases/residues, and output schema. Technical zero labels created by
`prepare-input` only preserve deterministic legacy cache keys; prediction never
treats them as observed negatives.

## Training and evaluation

The unified CLI forwards to the exact validated implementation:

```bash
python main.py train --config configs/m4_holo20k.yaml
python main.py eval --help
python main.py build-tensor-cache --help
python main.py build-graph-cache --help
python main.py build-surface-cache --help
python main.py build-static-cache --help
```

For the Holo20k experiment:

1. follow [`docs/DATASET_BUILD_CHECKLIST.md`](docs/DATASET_BUILD_CHECKLIST.md);
2. preserve all-partner residue-union events and deterministic case IDs;
3. enforce the registered length/cluster/distribution constraints;
4. require the GPSFun, Pinder-S apo, and complete Pinder-Large apo
   MMseq30/cov80 leakage audit to pass;
5. build all four cache families and run `scripts/audit_cache.py`;
6. train with `configs/m4_holo20k.yaml`; and
7. evaluate GPSFun, Pinder-S apo, and Pinder AF2 without changing test sets.

The registered v9 manifest has 20,000 unique structures, 9,812 clusters,
cluster cap 20, and zero MMseq30/cov80 hits. Its exact public artifact URL
remains unset until the validated manifest/audit bundle is published; see
[`assets/datasets.json`](assets/datasets.json).

## Reproducibility and validation

```bash
python -m pytest -q
python scripts/verify_sources.py
```

Five critical implementation files are pinned by SHA-256 and checked in CI.
Local tests, tracked-only installation, source hashes, and strict checkpoint
loading pass. The v0.1.0 code/checkpoint bundle is a prerelease for inference
and external validation. Same-GPU old/release logit equivalence, deterministic
one-step training equivalence, and publication of the exact dataset artifact
remain required before claiming exact end-to-end scientific reproduction.

See [`docs/VALIDATION.md`](docs/VALIDATION.md),
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), and
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the complete audit trail.

## License

Code and model weights are released under
[CC BY-NC-SA 4.0](LICENSE). Cite the associated manuscript when using M4 in
academic work; formal citation metadata will be added when the manuscript
record is available.
