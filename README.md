# GSBind: Gaussian surface–residue protein binding prediction

This repository contains the frozen E11 backbone, binding/SSM evidence, Gaussian-surface analyses, and the **2026-10-22 replacement of the downstream biological-information readouts**. The active downstream tasks are **AlloBench, AM and Src only**. Old 155D combined probes, source-to-target transfer readouts and KRAS result tables have been removed, not maintained as alternative current results. Raw experimental sources remain for attribution and label reconstruction.

## Current downstream results

| Probe | AlloBench AUPR | AM AUPR | Src Spearman ρ |
|---|---:|---:|---:|
| GSBind score, 1D | 0.074599 | 0.548396 | 0.388337 |
| Structure-only, 26D / 29D | 0.076044 | 0.647281 | 0.401681 |
| **GSBind centered hidden, 128D** | **0.154688** | **0.654665** | **0.517147** |
| PeSTo score, 1D | 0.080793 | 0.539383 | 0.146099 |

AlloBench/AM use **fixed optimizer iteration 50**; Src uses **fixed conjugate-gradient iteration 10**. Converged runs keep their last available iterate. These are not neural-network epochs, not the best checkpoint within a budget, and not loss-selected checkpoints. PeSTo uses its scalar binding score, never a hidden representation. All four methods use the same held-out rows and iteration budget for a task. Classification AUPR is trapezoidal precision–recall area, protein → component macro; Src is case/interaction-macro Spearman over complete out-of-fold predictions.

**Reporting boundary:** iteration budgets were chosen after exploratory inspection of held-out results. Fixed endpoints remove per-fold checkpoint cherry-picking but do not retroactively create an independent test set. Each task is trained separately; Src is within-system position generalization, not new-protein or allosteric-source transfer.

## Run from a public clone

Use Python 3.12 and the tested dependencies in `requirements-downstream.txt` for exact readout replay/refitting. The broader backbone package uses `pip install -e '.[test]'`; then install the downstream pins. Only load trusted repository joblib files.

```bash
pip install -e '.[test]'
pip install -r requirements-downstream.txt
python main.py downstream --action replay --output runs/downstream_replay
python main.py downstream --action train --output runs/downstream_refit
python main.py summarize --task all --output runs/public_recompute
python main.py verify
python main.py figures
python -m pytest -q
python tools/check_publication.py
```

Both replay and refitting use **only files in this checkout**: all probe feature rows, raw position labels, original fold assignments, per-fold regularization choices, 60 selected heads, and fixed-iteration reference predictions are included. Refitting performs no new validation split; it reuses the recorded training-side regularization choices, never held-out metrics. `--task AlloBench`, `--task AM` and `--task Src` restrict a downstream run. New outputs go to `runs/` unless explicitly directed elsewhere. Matching fixed-iteration predictions is checked automatically.

## Engineering map

- `src/gaussian_direct/`: atom encoder, Gaussian surface/residue forward model, preprocessing and Lightning backbone training.
- `src/gsbind/downstream.py`: four matched readouts, purged Src splits, fixed-iteration fitting, checkpoint replay, metrics and plotting.
- `configs/downstream_fixed.json`: authoritative task, budget and selection contract.
- `datasets/downstream/{AlloBench,AM,Src}/`: complete public feature chunks, row maps, labels, folds and numerical references.
- `checkpoints/downstream/`: 3 tasks × 4 methods × 5 folds, each with `head.joblib` and `SELECTION.json`.
- `results/downstream/`: authoritative metrics, per-fold metrics, per-protein/interaction metrics, complete OOF predictions and validation.
- `results/main/`, `results/ssm220/`, `results/figure3_*/`: unchanged backbone evidence and Gaussian-surface analyses.
- `docs/DOWNSTREAM_FIXED_PROTOCOL_CN.md`: exact methods, data, feature definitions and manuscript-ready replacement text.
- `docs/MAIN_TEXT_AUDIT_CN.md`: numerical audit, metric-name caveat and remaining manuscript/public-data limitations.
- `provenance/`, `FILE_INVENTORY.csv`, `DELIVERY.json`: source hashes, execution audit and public content inventory.

## Backbone evidence and release boundary

Binding manifests retain 20,801 training chains, 988 holo test cases and 874 apo/AF2 cases. Existing binding/SSM CSV columns named `aupr` / `aupr_mixed` implement **average precision (AP)**, not trapezoidal AUPR. Their numeric values are preserved; do not copy their legacy names as a definition of the new downstream metric. Main case-macro AP is 0.791500 / 0.728115 for E11 on holo / apo-AF2.

Frozen-backbone CPU examples and saved-prediction recomputation are public. Full backbone retraining and regeneration of all structure/surface features still require external bulk inputs; the old bulk archive has no registered public URL. This limitation does **not** affect the complete public downstream readout training/replay inputs. See `data_release/README.md`.

Source data retain their original attribution and redistribution conditions; repository inclusion does not change their licenses. See `LICENSE`.
