# Figure 3: paired binding-site reliability (2026-10-21)

Both `figures/GSBind_figure3_binding_v20261021.{pdf,png}` (nine panels) and `figures/GSBind_figure3_binding_core6_v20261021.{pdf,png}` (six panels) are included. The six-panel layout uses full-figure A/C/D/E/F/H, relabeled a–f; no estimates change between layouts. Previous releases are preserved.

This directory contains analysis CSVs, the validation contract, test records and plotting inputs. Protocol and results are in `../../docs/FIGURE3_BINDING_RELIABILITY_PROTOCOL_CN_20261021.md` and `../../docs/FIGURE3_BINDING_RELIABILITY_RESULTS_CN_20261021.md`.

The dense revision adds system distributions, model-by-budget mean/IQR grids, the complete 10%–90% reference-recall cost curve, spatial recall/area encoding and all-system joint benefits. See `../../docs/FIGURE3_COMMENT_RESPONSE_AND_DENSE_DESIGN_CN_20261021.md` for the original-figure comparison and seven-item comment response. The previous structural example remains supplementary, not evidence of superiority over all methods.

Replot from the repository root:

```sh
Rscript tools/plot_figure3_binding_reliability.R
```

Full residue-level source data, panel-organized CSVs, sensitivity outputs and a standalone plotting script are maintained in the `Lim-ZQ/GSBind_source_data` repository under `source_data/fig3_binding_v20261021/` and `GSBind_fig3_binding_v20261021/`. Full recomputation uses `tools/build_figure3_binding_reliability.py`, the original dataset/structure inputs, and the sibling `GSBind_source_data` checkout for cluster-sensitivity metadata. No scores or model weights are retrained.
