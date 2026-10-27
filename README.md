# GSBind: Gaussian surface–residue protein binding prediction

The current manuscript follows four figures: **overview → native binding prediction → Unbound localization → biological information increment**. The backbone E11 checkpoint is selected on the GPSFun AF2 full375 validation set after a complete E0–E19 sweep. Biological readouts use AlloBench source heads transferred to AM and separately supervised Src within-system probes. AM has no separately trained head or target-label fitting. Exact selection protocols are documented in `docs/GPSFUN_AF2_VALIDATION_CHECKPOINT_SELECTION_CN.md` and `docs/DOWNSTREAM_SELECTED_PROTOCOL_CN.md`.

## Current results

| Probe | AlloBench selected OOF AUPR | AM frozen-transfer AUPR | Src selected OOF Spearman ρ |
|---|---:|---:|---:|
| GSBind score, 1D | 0.074599 | 0.509336 | 0.388337 |
| Structure-only, 26D / 29D | 0.077680 | 0.648387 | 0.400463 |
| **GSBind centered hidden, 128D** | **0.159236** | **0.693699** | **0.512691** |
| PeSTo score, 1D | 0.080793 | 0.485176 | 0.146099 |

AlloBench hidden checkpoint iterations are **28/49/69/44/74**, chosen separately on the five source held-out folds over the original full optimizer history. These frozen source heads are applied directly to AM. Target components related to AlloBench use only source heads whose training folds exclude that component; entirely new components use the mean of all five heads. **0.693699 is this routed all-position AM result, not an unrestricted all-five ensemble.** Source-fold selection exposure can remain for related components. The supplementary source-component-disjoint AM subset has 371 positions, 26 proteins and 25 components; hidden AUPR is **0.726231**. That is a different evaluation subset, not a replacement full-AM score.

Src hidden iterations are **1/7/2/4/7**, chosen by each internal held-out fold's Spearman correlation. Corresponding FL/KD physical positions share folds, with 10-position blocks and ±5-position training exclusion. The reported 0.512691 is complete out-of-fold case/interaction-macro Spearman, not the mean of five fold correlations.

**Selection boundary:** AlloBench and Src checkpoints are explicitly selected using their source held-out/test folds. Their reported source OOF scores are exploratory test-selected scores, not independent test estimates. AM labels do not fit or select heads, but full AM is not completely source-independent because of shared source components. PeSTo uses only its scalar binding score, never hidden features. Classification AUPR is trapezoidal precision–recall area, protein → homology-component macro.

**Backbone validation:** E11 was selected only on GPSFun AF2 full375 (375 proteins, 77,978 valid residues, 11,350 positives) after evaluating all E0–E19 checkpoints. The primary score equally weights population-z-scored pooled, case-mean, case-median and macro-recall families; validation loss and Brier are excluded. This validation set is distinct from the final Bound/Unbound and downstream tests.

## Public reproduction

```bash
pip install -e '.[test]'
pip install -r requirements-downstream.txt
python main.py downstream --action replay --output runs/selected_replay
python main.py downstream --action train --output runs/source_search_reconstruction
python main.py summarize --task all --output runs/public_recompute
python main.py --config configs/main.json select-checkpoint
python main.py verify
python main.py figures
python -m pytest -q
python tools/check_publication.py
```

All input features, raw position labels, original folds, AM routing, **40 selected source heads**, source candidate-metric histories, historical training-reference heads and selected prediction references are included. Python 3.12 and tested package pins reproduce the current numerical trajectory. Load only trusted joblib files.

`--action replay` performs inference only. `--action train` reconstructs the source optimizer searches and best-test selection, **reusing the historical converged reference heads as the original candidate set requires**; it is not a from-scratch reconstruction of those historical fits or a new hyperparameter search. AM is never trained: `--task AM --action train` reconstructs AlloBench source searches, then transfers. `--task AlloBench`, `--task AM` and `--task Src` restrict evaluation. Outputs go to `runs/` unless explicitly redirected; selected-reference checks fail explicitly on numerical mismatch.

## Engineering map

- `src/gaussian_direct/`: frozen E11 model, atom encoder, Gaussian-surface forward, preprocessing and backbone training.
- `src/gaussian_direct/validation.py`: post-training GPSFun AF2 full375 sweep and performance-only E0–E19 checkpoint selection.
- `src/gsbind/pinder_pipeline.py`: complete public-PINDER API rebuild of the E11 training pool, official held-out test manifest, coordinate labels, holo unions, side-cluster leakage cleaning and cap5 selection.
- `src/gsbind/downstream.py`: selected-source validation, AM source-component routing, frozen replay, metrics and plots.
- `src/gsbind/downstream_training.py`: reconstruction of AlloBench/Src optimizer histories and source-test selection; no AM fitting.
- `configs/downstream_selected.json`: sole active downstream protocol.
- `datasets/downstream/{AlloBench,AM,Src}/`: complete features, labels, folds and prediction references; `AM/TRANSFER_ROUTING.csv` records eligible heads and independent subset.
- `checkpoints/downstream/{AlloBench,Src}/`: 2 sources × 4 probes × 5 folds, each with selected head, selection metadata, candidate history and historical reference head. There is no AM checkpoint directory.
- `results/downstream/`: primary and supplementary metrics, source-fold metrics, detailed unit metrics and per-task `predictions.csv.gz`.
- `figures/downstream_selected_transfer.*`: AlloBench/AM component violins and a Src construct comparison; full plot data in `figures/source_data/downstream_selected/`.
- `figures/downstream_results_four_panel.*`: restyled three-result panels plus pooled-AUPR bars; individual panels are `downstream_A_AlloBench.*`, `downstream_B_AM.*`, `downstream_C_Src.*`, and `downstream_D_pooled_AUPR.*`.
- `figures/downstream_pooled_pr.*`: actual pooled precision–recall curves and trapezoidal AUPR; exact curve coordinates and thresholds in `figures/source_data/downstream_pooled_pr/`.
- `figures/downstream_task_biology.*`, `figures/downstream_task_workflow.*`: editable conceptual task diagrams; definitions and caveats in `docs/DOWNSTREAM_TASK_EXPLANATION_CN.md`.
- `figures/structural_cases/`: PyMOL-rendered ABL1 allosteric-pocket and Src activation-segment / functional-readout cases, residue maps and editable sessions; see `docs/STRUCTURAL_READOUT_CASES_CN.md`.
- `docs/DOWNSTREAM_SELECTED_PROTOCOL_CN.md`: methods, data, manuscript-ready English text and figure overview.
- `docs/PINDER_TRAINING_DATA_REBUILD_CN.md`: reproducible official-API E11 training/test rebuild protocol; run `python main.py rebuild-pinder` and `python main.py validate-pinder-splits`.
- `docs/downstream_references.bib`, `docs/PAPER_EVIDENCE_ALIGNMENT_CN.md`: references and current manuscript/evidence mapping. The previous audit is archived under `history/pre_alignment_20261023/docs/`.

## Unchanged backbone evidence

Binding manifests retain 20,801 training records, 988 Bound and 874 Unbound test cases. Unbound combines experimental unbound and AF2 structures. Legacy binding/SSM fields named `aupr` / `aupr_mixed` implement **average precision (AP)**, not trapezoidal AUPR; numbers remain unchanged. E11 case-macro AP is 0.791500 / 0.728115 on Bound / Unbound; these are not the manuscript's pooled AP values. Current Figure3 is the coordinate-valid conformational/localization analysis in the figure-data repository, not the older pathway-intervention Figure3. Historical intervention code and data remain available but do not supply current Figure3 claims.

This engineering tree supports current readout inference/search reconstruction and saved-prediction recomputation. Full backbone training can reconstruct the PINDER pool from the official public API with `python main.py rebuild-pinder`; bundled historical bulk caches remain available through `data_release/README.md`. The companion manuscript uses the aligned four-figure story; `docs/PAPER_EVIDENCE_ALIGNMENT_CN.md` maps its figure sources and bundled hashes. Final author metadata and venue approval remain manuscript responsibilities.

Original data retain source attribution and redistribution conditions. See `LICENSE`.
