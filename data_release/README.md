# Bulk data release

## 2026-10-23 downstream replacement

The current **AlloBench source-test-selected heads → frozen AM transfer**, and **Src internal five-fold best-test-selected heads**, are fully self-contained in Git. Complete features, labels, source folds, AM component routing, 40 selected source heads, candidate histories and 40 historical reference heads are included. Replay needs no archive. `--action train` reconstructs the two source optimizer searches using historical reference candidates; it never fits AM and is not a from-scratch reconstruction of the historical baseline fits. The archive is a historical bulk-input source for full backbone regeneration, not the current readout release. Do not restore obsolete results or checkpoints from it.

## Current status: public download URL pending

The server has a complete, checksum-verified bundle. This Git repository contains the code, E11 weights, readout weights, data manifests/labels, final CSV metrics, detailed saved predictions and CPU examples. It deliberately does not commit the bulk training structures/caches or large allosteric hidden tensors.

The GPSFun AF2 full375 checkpoint-selection contract is in `gpsfun_af2_full375_validation_contract.json`: 375 proteins, 77,978 valid residues, 11,350 positives, strict sequence-aligned AF2 inputs, and E0–E19 performance-only family-balanced z-score selection. The compact NPZ cache remains external bulk data; `python main.py select-checkpoint` requires the fixed manifest and cache installed at the paths in `configs/main.json`.

| Artifact | Status |
|---|---|
| `GSbind_paper_20261020.tar.zst` | Exists on the project server; public URL **not yet assigned** |
| Archive bytes | `15323513322` |
| Archive SHA256 | `0bbbc666e9f5ad74ae9dc273642f788c4bf676b3673a2446d0a67c470cfa8c65` |
| Download registration | `bundle.json`, currently `url: null` |
| Files absent from Git | `EXTERNAL_FILES.csv`, exact paths, sizes and SHA256 |
| Full server snapshot | `FULL_BUNDLE_INVENTORY.csv`, 61,801 files; not the public checkout inventory |
| Public checkout inventory | Root `FILE_INVENTORY.csv`; excludes itself and generated `runs/` |

The archive's root directory is `GSbind/`. It is an older snapshot, retained as the source of unchanged bulk datasets; it does not contain the current all-position functional analysis. Install **only missing datasets**, preserving current public files, including the updated functional caches:

```bash
sha256sum GSbind_paper_20261020.tar.zst
mkdir -p /path/to/temporary_extract
tar --zstd -xf GSbind_paper_20261020.tar.zst -C /path/to/temporary_extract
rsync -a --ignore-existing /path/to/temporary_extract/GSbind/datasets/ /path/to/Guassion-PPBS/datasets/
cd /path/to/Guassion-PPBS
python tools/check_publication.py --external
python main.py verify
python main.py evidence --task all
```

`check_publication.py --external` requires every registered external file and checks its bytes/hash; it fails rather than silently accepting a partial download. Do not overwrite root `FILE_INVENTORY.csv` with the full-bundle inventory: they describe different packages.

## Which operations work without bulk data?

| Operation | Public clone | Bulk data required |
|---|---|---|
| Read all maintext metrics and detailed saved predictions | Yes | No |
| Recompute metrics from saved predictions: `main.py summarize` | Yes | No |
| CPU example inference, evaluation and Lightning gradient step | Yes | No |
| Current selected-source replay, source search reconstruction and frozen AM transfer | Yes | No |
| Main/SSM evidence from saved binding scores | Yes | No |
| Regenerate all frozen features from original structures | No | Yes |
| All-case surface forward and complete model training | No | Yes |
| Rebuild SSM labels from original large assay tables | No | Yes |

## 2026-10-27 PINDER API alternative

The original PINDER index and native holo structures can now be rebuilt directly from the official `pinder` 2024-02 API. This avoids the older private bulk bundle for backbone data construction, although it is slower and downloads the public source files anew. The complete protocol is `docs/PINDER_TRAINING_DATA_REBUILD_CN.md` and the CLI is `python main.py rebuild-pinder`.

Once a public hosting location is selected, update `bundle.json` and this page with the real URL. Large upstream archives and experimental source files remain subject to their original licensing and attribution.
