# Gaussian-direct128 architecture

Gaussian-direct128 maps protein atomic structure to residue-level binding probabilities through two coupled stages: atom encoding and Gaussian surface interaction.

## Forward path

1. `AtomEncoder` converts atom features and local topology into scalar/vector atom states.
2. Atom states are pooled into residue states and projected to the Gaussian representation.
3. A per-residue Gaussian field projects residue context onto sampled surface points.
4. Pair-Gaussian corrections integrate cooperative residue context.
5. GVP graph encoders update surface states at multiple graph scales.
6. Connected patch pooling and surface-to-residue attention exchange information.
7. Three co-update rounds refine residue and surface states.
8. A direct head emits protein, nucleic-acid, ion, ligand, and lipid channels.

The released E11 checkpoint uses `architecture=vg`, `gaussian_dim=128`, and `direct_protein_head=true`.

## Module map

| Path | Responsibility |
|---|---|
| `model.py` | `GaussianDirectModel` and Gaussian interaction block |
| `atom_encoding/atom_encoder.py` | Atom feature, state update, pooling, and residue decoder |
| `atom_encoding/dataset.py` | Compact NPZ training payload |
| `atom_encoding/surface_cache.py` | Surface sampling and graph cache |
| `modules/gaussian_projection.py` | Per-residue adaptive Gaussian field |
| `modules/pair_gaussian.py` | Residue-pair Gaussian correction |
| `modules/surface_encoder.py` | GVP surface encoder and connected patches |
| `modules/cross_scale.py` | Surface-to-residue attention |
| `modules/global_context.py` | Residue and surface global-context feedback |
| `modules/semantic_patch.py` | Optional semantic patch objective |
| `lightning.py` | Deterministic distributed training loop |
| `app.py` | Inference and evaluation use cases |

## Parameter budget

The released model contains 3,355,001 parameters: 1,474,957 in `AtomEncoder` and 1,880,044 in the Gaussian/direct path. Bypass and parameter-matched switches exist in the implementation; their presence alone is not a reported retraining experiment. The current paper engineering contains the E11 model and Gaussian-surface pathway interventions.
