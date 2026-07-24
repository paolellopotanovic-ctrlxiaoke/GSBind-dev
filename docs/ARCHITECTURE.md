# Architecture and source-reading guide

M4 is organized by scientific responsibility rather than by experiment history.
The public implementation is under `src/m4`; `legacy_repo` is a frozen
numerical oracle and is not the default model or training path.

## Start here

Read these files in order:

1. `src/m4/models/m4.py` defines the public build/load/batch-forward contract.
2. `src/m4/models/components/predictor.py` assembles the complete M4 network.
   Its constructor is the architecture composition root.
3. `src/m4/models/components/execution.py` contains the complete forward data
   flow. Parameter-free helper stages live in `components/runtime/`.
4. `src/m4/losses.py` defines the weighted multi-objective training target.
5. `src/m4/data/datamodule.py` owns manifests, grouped splits, class weighting,
   and strict four-cache loaders.
6. `src/m4/lightning/module.py` owns optimization, scheduling, epoch state,
   logging, and checkpoint export.
7. `src/m4/predictor.py` is the label-free inference application that maps PDB
   residue identities to final, prior, and surface-conditioned probabilities.

## Model decomposition

```text
src/m4/models/
|-- m4.py                         public build/load/forward contract
|-- config.py                     checkpoint args -> constructor kwargs
`-- components/
    |-- residue_encoder.py        atom pooling, multiview GVP, Gear/OAGNN refiners
    |-- surface_encoder.py        surface GVP, connected patches, Laplacian blocks
    |-- gaussian_projection.py    single-residue and multiscale Gaussian fields
    |-- pair_gaussian.py          pair/triplet-conditioned Gaussian corrections
    |-- cross_scale.py            residue<->surface attention and patch feedback
    |-- predictor.py              architecture assembly
    |-- execution.py              end-to-end forward algorithm
    `-- runtime/
        |-- stochastic.py         feature/view/stochastic helpers
        |-- surface_flow.py        residue/surface graph runtime
        `-- gaussian_flow.py       chunked pair-Gaussian runtime
```

The high-level path is:

```text
atoms -> residue features -> multiview residue graph encoder
      -> Gaussian residue-to-surface field -> surface GVP/patch processing
      -> surface-to-residue fusion -> semantic patch refinement
      -> per-residue interface logits
```

All component classes retain the validated module attribute names. Therefore the
released checkpoint still has 791 state tensors and 4,360,516 parameters and
loads with `strict=True` without key translation.

## Training boundaries

The paper-facing training path is PyTorch Lightning:

```bash
python main.py train --config configs/m4_holo20k.yaml
```

- `M4DataModule` performs state filtering, fixed/grouped splits, positive-class
  weighting, and cache-backed loading.
- `M4LightningModule` performs forward, exact losses, AdamW, warmup/cosine or
  linear scheduling, AMP, accumulation, clipping, metrics, and checkpointing.
- Epoch seeding and the semantic-boundary activation epoch match the validated
  trainer.
- Lightning checkpoints also export the original `model_state_dict` and
  `args` fields so released inference remains compatible.

The original trainer is intentionally available only as an oracle:

```bash
python main.py train-legacy --config configs/m4_holo20k.yaml
```

It must not receive new model logic. Changes belong in the modular path and must
pass the equivalence audit.

## Equivalence contract

`tests/test_modular_training.py` verifies identical constructor state, parameter
keys, tensor shapes, strict cross-loading, loss behavior, and epoch-state logic.
For publication validation, run `scripts/validate_modular_equivalence.py` on a
CUDA machine with one real cache-backed batch. It compares:

- eval and train outputs;
- every objective component;
- every parameter gradient; and
- one AdamW update.

The frozen files in `src/m4/legacy_repo` remain SHA-256 pinned so the oracle
cannot drift while the public architecture evolves.
