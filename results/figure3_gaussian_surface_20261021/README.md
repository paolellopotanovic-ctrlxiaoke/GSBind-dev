# Gaussian surface pathway Figure 3

Parallel six-panel revision, not a replacement of prior Figure 3 files.

- A: system-paired native / half / zero Gaussian surface-feedback dependence.
- B: observed 0/50/100% attenuation responses at the surface-feedback and pair-message sites.
- C: high-score-local pair removal versus same-count random removal; jointly reports AP damage, correct-candidate loss and positive/background priority redistribution.
- D: matched holo-to-unbound binding accuracy, all four methods on common reference positions.
- E: cross-state overlap versus retained correct interface candidates, not stability alone.
- F: disclosed 4ml9 AF2 R illustrative example, same surface/view/score scale and aligned residue-rank effects.

`figures/GSBind_figure3_gaussian_surface_v20261021.pdf` and `.png` are the complete figure; A–F are also exported separately. Tables retain all 1862 inputs and all 12 original intervention conditions. Main means are system-equal; random seeds average within case. Transfer has 788 uniquely mapped cases and a distinct fixed-holo reference contract.

Protocols and numerical interpretation are in `../../docs/FIGURE3_GAUSSIAN_SURFACE_PROTOCOL_CN_20261021.md` and `../../docs/FIGURE3_GAUSSIAN_SURFACE_RESULTS_CN_20261021.md`. Validation, independent local-mask checks, saved-data replay and tests are included here.

Replot from the engineering repository root:

```sh
python tools/plot_figure3_gaussian_surface.py
```

Recompute metrics from these saved results without a GPU or hidden-cache dependency, writing a separate output directory:

```sh
python tools/build_figure3_gaussian_surface.py \
  --source-dir results/figure3_gaussian_surface_20261021/raw \
  --paired-dir results/figure3_gaussian_surface_20261021 \
  --output-dir /tmp/gsbind_surface_metric_replay
```

The full 1862-case geometry audit additionally uses the original per-case inference cache, not required for the figure replot or metric replay. Its complete row-level audit is in `local_intervention_mask_audit.csv`. Source-data repository delivery uses `source_data/fig3_gaussian_surface_v20261021/`, with sibling `src/` and `fig/` directories under `GSBind_fig3_gaussian_surface_v20261021/`; raw/ includes full saved residue probabilities and the selected example's actual output packet.
