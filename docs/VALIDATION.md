# Release validation

Validation date: 2026-07-24.

## Completed

- The five critical M4 sources match the recorded SHA-256 digests.
- All release Python files compile.
- The tracked-only Git archive installs with `pip install -e . --no-deps`.
- Top-level, training, inference, tensor-cache, and graph-cache imports parse in
  the server environment.
- Public model and dataloader imports resolve to the release snapshot.
- Holo20k v9 epoch 4 checkpoint loads strictly:
  - parameters: 4,360,516
  - state tensors: 791
  - missing keys: 0
  - unexpected keys: 0
- Local unit tests pass: CLI/config, source hashes, label-free input and
  prediction contracts, registries, union/replacement/distribution logic, and
  checksum-guarded ordinal repair and the one-command inference plan (22 tests).
- Label-free `prepare-input` successfully encodes the included 46-residue 1CRN
  example in strict residue order.
- The checkpoint registry pins the immutable v0.1.0 GitHub Release asset URL,
  size, and SHA-256. The exact dataset URL remains intentionally unset until
  that separate artifact is retrieved and published.
- Full Pinder holo construction, all-partner union, deterministic replacement,
  distribution audit, and fixed MMseq30/cov80 audit scripts are included.
- A staged tracked-only archive builds and installs as `m4-interface-0.1.0` with
  modern Python/setuptools without reading files outside the archive.

## Validated environment

- Python 3.10.8
- PyTorch 2.1.2+cu121
- CUDA runtime 12.1
- NumPy 1.26.3
- SciPy 1.15.3
- scikit-learn 1.7.2
- Gemmi 0.7.5
- h5py 3.16.0
- pandas 2.3.3
- PyYAML 6.0.1

## Remaining release gate

The validation server reported `torch.cuda.is_available() == False`. A CPU
end-to-end pass was stopped after ten minutes because M4 processes 1,024 surface
points even for the selected 46-residue regression case. Before tagging a public
release, run `scripts/validate_server_equivalence.sh` on an available CUDA GPU
and require byte-identical old/new evaluation artifacts, followed by one
deterministic training step using the same GPU and software image.

The v0.1.0 checkpoint/code bundle is published as a prerelease for inference
and external validation. Exact end-to-end scientific reproduction must not be
claimed until the CUDA equivalence gate passes and the immutable dataset asset
is published.
