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
the validated research implementation while exposing a modular PyTorch
Lightning codebase for dataset construction, caching, training, evaluation,
and label-free inference.

> **Scope.** M4 predicts protein-protein interface residues. It is a
> prioritization model, not a docking engine, affinity predictor, or causal
> mechanism detector.

## Start here: understand M4 in 15 minutes

M4 has one public scientific path. Start with these files in this order:

| Question | Open this file | What it contains |
|---|---|---|
| What enters and leaves the model? | [`src/m4/models/m4.py`](src/m4/models/m4.py) | Public build, strict checkpoint-load, and batch-forward contract |
| Which scientific modules make up M4? | [`src/m4/models/components/predictor.py`](src/m4/models/components/predictor.py) | Architecture composition root: residue encoder, Gaussian field, surface encoder, fusion, and heads |
| What happens in one forward pass? | [`src/m4/models/components/execution.py`](src/m4/models/components/execution.py) | Complete operation order from atoms to residue logits |
| How is the Gaussian surface field computed? | [`gaussian_projection.py`](src/m4/models/components/gaussian_projection.py) and [`pair_gaussian.py`](src/m4/models/components/pair_gaussian.py) | Single-residue projection and pair-conditioned correction |
| How are residue and surface states exchanged? | [`cross_scale.py`](src/m4/models/components/cross_scale.py) | Residue-to-surface, surface-to-residue, patch, and feedback operators |
| What objective is optimized? | [`src/m4/losses.py`](src/m4/losses.py) | Complete weighted training objective |
| Where is the training loop? | [`src/m4/lightning/module.py`](src/m4/lightning/module.py) | Lightning steps, optimizer, exact scheduler, metrics, and checkpoint export |
| How are samples and caches loaded? | [`src/m4/data/datamodule.py`](src/m4/data/datamodule.py) | Manifest splits, class weighting, and strict four-cache loaders |
| How is an unlabeled PDB predicted? | [`src/m4/predictor.py`](src/m4/predictor.py) | Label-free inference and residue-level CSV export |

A single forward pass follows five scientific stages:

```text
PDB atoms
   -> multi-view residue graph encoding
   -> residue-conditioned Gaussian field on 1,024 surface points
   -> surface graph and connected-patch reasoning
   -> bidirectional residue/surface fusion
   -> one protein-interface probability per residue
```

The main forward algorithm stays in one file so its operation order is
auditable. Parameter-free helpers are separated by responsibility under
[`src/m4/models/components/runtime/`](src/m4/models/components/runtime):
feature/stochastic transforms, surface flow, and chunked pair-Gaussian flow.
`src/m4/legacy_repo` contains the frozen numerical oracle and the dependency
closure for low-level historical operators/cache formats. It is not a public
training entry: architecture assembly, forward orchestration, losses, data,
Lightning training, and inference all enter through `src/m4`.

For a deeper method-to-code map, continue with
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) after reading the three primary
files above.

## Repository layout

```text
main.py                     unified train/eval/cache/data-entry CLI
inference.py                label-free per-residue prediction CLI
src/m4/                     modular model, data, loss, Lightning, and inference code
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
- At least 130 GB of free disk for the complete Holo20k structure and cache
  build. The validated cache occupied approximately 105 GB.

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

The unified CLI trains the modular implementation through PyTorch Lightning:

```bash
python main.py train --config configs/m4_holo20k.yaml
python main.py eval --help
python main.py build-tensor-cache --help
python main.py build-graph-cache --help
python main.py build-surface-cache --help
python main.py build-static-cache --help
```

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the source-reading
order. The complete model is assembled in
`src/m4/models/components/predictor.py`; its runtime graph is in
`src/m4/models/components/execution.py`. The frozen historical trainer remains
available as `python main.py train-legacy --config ...` solely for numerical
equivalence audits.

For the Holo20k experiment:

Download the immutable paper manifest, hydrate its 20,000 public Pinder
structures, and replace only the obsolete server path column:

```bash
python scripts/download_dataset.py holo20k_v9_mmseqclean --include-audit
python scripts/prepare_holo20k.py \
  --manifest data/manifests/holo20k_gpsfun_clusterdiverse_v9_mmseqclean_train.csv \
  --pdb-dir data/pinder/2024-02/pdbs \
  --output-manifest data/manifests/holo20k_v9.local.csv \
  --workers 32
```

`prepare_holo20k.py` verifies the immutable manifest SHA-256, downloads each
structure from the Pinder 2024-02 public bucket, applies the one registered
fixed-width PDB B-factor repair only when the source hash matches, and records
the source and localized manifest hashes. Labels, masks, residue ordinals,
clusters, row order, and case IDs are unchanged.

Then build the four cache families using the commands in
[`docs/DATASET_BUILD_CHECKLIST.md`](docs/DATASET_BUILD_CHECKLIST.md), require a
zero-error cache audit, update the four cache paths and manifest path in
`configs/m4_holo20k.yaml`, and run:

```bash
python main.py train --config configs/m4_holo20k.yaml
```

The same checklist documents the longer raw-Pinder reconstruction route. That
route is useful for auditing dataset logic; the immutable Release manifest is
required for byte-identical sample selection.

The registered v9 manifest has 20,000 unique structures, 9,812 clusters,
cluster cap 20, and zero MMseq30/cov80 hits. The exact manifest and its
construction/MMseq2/cache audit bundle are immutable v0.1.0 GitHub Release
assets registered in [`assets/datasets.json`](assets/datasets.json).

## Reproducibility and validation

```bash
python -m pytest -q
python scripts/verify_sources.py
```

Five frozen oracle files are pinned by SHA-256 and checked in CI. Local tests,
tracked-only installation, source hashes, strict checkpoint loading, and exact
dataset artifact identity pass. A real two-case RTX 4080 audit with the Holo20k
E4 checkpoint produced identical eval outputs, train outputs, and all 14 loss
components; the worst gradient difference was `5.22e-8` and the worst
post-AdamW parameter difference was below `2e-6`. The machine-readable audit is
registered in [`docs/VALIDATION.md`](docs/VALIDATION.md).

See [`docs/VALIDATION.md`](docs/VALIDATION.md),
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md), and
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the complete audit trail. The
exact data identity, registered PDB repair, cache sizes, and audit-bundle
contents are documented in
[`docs/DATASET_RELEASE.md`](docs/DATASET_RELEASE.md).

## License

Code and model weights are released under
[CC BY-NC-SA 4.0](LICENSE). Cite the associated manuscript when using M4 in
academic work; formal citation metadata will be added when the manuscript
record is available.
