# GSbind

**Gaussian-surface interaction for protein binding-site prediction and functional representation analysis.**

This is the current paper engineering for **Gaussian-direct128 E11 / GSbind**, not a collection of historical experiments. It contains the model, Lightning training pipeline, final evaluation protocols, E11 weights, frozen diagnostic readouts, paper metrics, and residue-level prediction tables.

旧仓库已完整保存在 [`archive/pre-gsbind-20261020`](https://github.com/paolellopotanovic-ctrlxiaoke/Guassion-PPBS/tree/archive/pre-gsbind-20261020) 分支及 `legacy-pre-gsbind-20261020` 标签。默认分支只维护当前论文主工程，见 [历史入口](HISTORY.md)。

## Paper scope and results

| Evidence | Current scope | Main result / entry |
|---|---|---|
| Binding-site prediction | 20,801 training records; official MMseqs holo 988 and APO/AF2 874 test cases | GSbind case-macro AUPR **0.7915 / 0.7281**; PeSTo **0.7380 / 0.6770** |
| Gaussian-surface pathway | Surface-to-residue feedback/fusion; native, identity, retain50, retain00 | Full cutoff AUPR **0.5220 / 0.4765** on the two binding panels |
| Allosteric representation | AlloBench383 and AM127 grouped nested fivefold; fixed-source transfers to four other sources and AM | Geometry / AA / hidden / full; component-macro AUPR **0.1697** AlloBench, **0.7216** AM for full |
| SSM220 | Native binding-score prioritization of experimental sensitivity | All 220 inputs retained; coverage differs by sensitivity endpoint; no hidden SSM classifier |
| KRAS/Src | Frozen diagnostic readouts versus experimental functional/folding effects | Low-score non-contact functional rho **0.4498 / 0.3634**, full surface cutoff **0.0223 / 0.1195** |

Allosteric probes use supervised linear logistic readouts on a **frozen** binding model, not retraining of GSbind. KRAS/Src retain their separately frozen functional readout protocol. Gaussian-surface cutoff is a model intervention, not a biochemical knockout or proof of an allosteric signal-transmission pathway.

## What is available now?

**Included in Git:** all source code, training/configuration scripts, E11 weights, four-method probe weights, functional readout weights, complete train/test manifests and labels, current result CSVs, detailed saved prediction Parquet files, figure source tables, small CPU examples, functional source measurements, and mapping/homology audits.

**External data URL pending:** bulk structures, training compact caches, allosteric seven-condition hidden tensors and per-protein forward caches, and large SSM raw assay tables. The complete server bundle exists (15.32 GB compressed); its checksum and exact missing-file inventory are in [`data_release/`](data_release/README.md). A normal Git clone is **not** the full training-data download. No working public download URL is claimed yet.

The available prediction tables already support recomputing paper metrics without downloading bulk inputs. Re-extracting all hidden states, refitting allosteric probes, and full training require the external data.

## Quick start: public checkout

Python 3.10 is the tested environment. Install PyTorch appropriate for your hardware first if needed.

```bash
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python tools/check_publication.py
python -m pytest tests -q
python main.py --config configs/smoke.json infer
python main.py --config configs/smoke.json evaluate
python tools/smoke_lightning.py
python main.py summarize --task all --output runs/public_recompute
python main.py figures
```

`summarize` recomputes metrics, paired contrasts and confidence intervals. Allosteric probabilities are read from the included saved prediction tables and checked against native reference predictions; this command does **not** replay hidden features or refit probes. Main/SSM use saved binding scores; KRAS/Src replay their frozen readouts from included small hidden arrays. Outputs go to `runs/` and do not overwrite the delivered paper tables.

`figures` creates basic evidence PDFs and refreshes source CSVs. These are not the final four-panel publication layout. [`FOUR_FIGURE_PLAN_CN.md`](docs/FOUR_FIGURE_PLAN_CN.md) specifies the final figure narrative.

## Full-data reproduction

After the external bundle is available, overlay its **`datasets/` directory** into this repository, retaining current public code/docs/results. Its archive directory is named `GSbind`, regardless of the name of this checkout. See [data installation and checksums](data_release/README.md).

```bash
python main.py verify
python main.py evidence --task all
python main.py rebuild-allosteric-labels
python main.py rebuild-ssm-labels
python main.py rebuild-functional-labels
python main.py fit-probes --cohort both --output runs/refit_probes
python main.py --config configs/main.json train
python main.py --config configs/main.json evaluate
python main.py --config configs/apo_af2.json evaluate
```

`evidence --task allosteric` replays frozen heads against the full hidden tensors. `fit-probes` trains only the four small logistic classifiers, using component-grouped outer5/inner3 folds and train-side regularization/threshold selection. The main-model checkpoint remains frozen. `train` is the separate Lightning main-task training entrypoint and saves new checkpoints under `runs/`.

## Engineering map

```text
main.py + configs/                 one application entrypoint and typed configurations
src/gaussian_direct/               model, atom encoder, Gaussian-surface, preprocessing
  lightning.py                    LightningModule + LightningDataModule + optimizer/loss
  app.py                          preprocessing / train / infer / evaluate orchestration
src/gsbind/
  allostery.py + probe_training.py  exact four-method readout and grouped evaluation
  evidence.py                     main / allosteric / score-only SSM / KRAS-Src endpoints
  forward.py + surface.py         actual Gaussian-surface forward interventions
  metrics.py + labels.py          shared metric, statistical, and label contracts
datasets/                         manifests, labels, prediction inputs, source audits
checkpoints/                      E11 and frozen diagnostic readouts
results/{main,allosteric,ssm220,functional}/
                                  authoritative final result CSVs and detailed predictions
figures/source_data/              CSV copies used by basic plots
data_release/                     bulk-data status and checksums; no fake download links
docs/ + tests/ + provenance/       methods, paper narrative, protocol tests and audit records
```

## Paper writing and plotting

- [完整论文证据报告](docs/PAPER_EVIDENCE_REPORT_CN.md)
- [架构说明](docs/ARCHITECTURE.md) · [工程与协议](docs/ENGINEERING_CONTRACT_CN.md)
- [指标和统计契约](docs/METRIC_CONTRACT_CN.md) · [数据字典](docs/DATA_DICTIONARY_CN.md)
- [四张主图设计](docs/FOUR_FIGURE_PLAN_CN.md) · [全部表格字段索引](docs/TABLE_INDEX.csv)
- [变构原始来源与重建](docs/ALLOSTERY_REPRODUCTION_CN.md)
- [主工程审查与公开范围](docs/PUBLICATION_AUDIT_CN.md)
- [当前变构AUPR主表](results/allosteric/MAIN_AUPR_TABLE.csv)

The main tables use **AUPR (average precision)**. MCC, AUROC and F1 are retained as supplementary metrics; KRAS/Src use continuous-effect Spearman correlation and fixed-budget enrichment. Case-macro, component-macro and pooled metrics must not be interchanged. Unknown experimental labels are not converted to negatives.

CSV tables are directly viewable. Larger residue-level predictions use Parquet; export them without loading all rows at once:

```bash
python tools/export_tables.py --input results/allosteric/AlloBench/predictions.parquet \
  --output runs/csv/AlloBench_predictions.csv.gz
```

Original data retain their source attribution and redistribution restrictions; repository inclusion does not change their licenses. The repository license is retained in `LICENSE`.
