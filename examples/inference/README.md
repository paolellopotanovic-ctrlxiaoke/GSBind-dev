# Minimal label-free inference example

`1CRN.pdb` is the 46-residue crambin structure downloaded from RCSB PDB. The
PDB archive distributes structure data under CC0. Its SHA-256 is
`42199a30a0701864a2a5cc76cd7f35cc544cd0e65fbcf63e03c166543249b811`.

Generate the internal cache manifest. No biological labels are supplied; the
generated zero labels are technical placeholders used only to keep legacy cache
keys deterministic.

Download and verify the registered checkpoint:

```bash
python scripts/download_checkpoint.py m4_holo20k_v9_clean_epoch4
python scripts/validate_checkpoint.py \
  checkpoints/m4_holo20k_v9_clean_epoch4.pt \
  --registry-name m4_holo20k_v9_clean_epoch4
```

```bash
python main.py prepare-input --pdb examples/inference/1CRN.pdb --case-id 1CRN \
  --input-state apo --output work/1CRN/manifest.csv
```

Build exactly the four inputs used by training:

```bash
python main.py build-tensor-cache --manifest work/1CRN/manifest.csv \
  --cache-dir work/1CRN/tensor --workers 1
python main.py build-graph-cache --manifest work/1CRN/manifest.csv \
  --tensor-cache-dir work/1CRN/tensor --graph-cache-dir work/1CRN/graph \
  --ks 16,24,32 --workers 1
python main.py build-surface-cache --manifest work/1CRN/manifest.csv \
  --tensor-cache-dir work/1CRN/tensor --residue-graph-cache-dir work/1CRN/graph \
  --output-dir work/1CRN/surface --states apo --surface-kind shrake \
  --surface-points 1024 --surface-graph-k 16 \
  --surface-cache-version s1_surface_shrake_v1
python main.py build-static-cache --manifest work/1CRN/manifest.csv \
  --tensor-cache-dir work/1CRN/tensor --residue-graph-cache-dir work/1CRN/graph \
  --surface-cache-dir work/1CRN/surface --output-dir work/1CRN/static \
  --surface-cache-version s1_surface_shrake_v1 \
  --geometry-cache-version e89_dense_geometry_v1 --surface-graph-k 32
```

The static-cache builder and practical M4 forward pass require CUDA. Then run:

```bash
python inference.py --manifest work/1CRN/manifest.csv \
  --checkpoint checkpoints/m4_holo20k_v9_clean_epoch4.pt \
  --output-dir work/1CRN/predictions --tensor-cache-dir work/1CRN/tensor \
  --residue-graph-cache-dir work/1CRN/graph \
  --s1-surface-cache-dir work/1CRN/surface \
  --s1-static-geometry-cache-dir work/1CRN/static
```

The result is `residue_predictions.csv`, one row per PDB residue. Inference
fails on any missing cache, skipped case, residue-order mismatch or non-strict
checkpoint load. Columns include chain ID, PDB residue number, insertion code,
amino acid, validity mask, final interface probability, prior probability, and
surface-conditioned probability. `prediction_metadata.json` records checkpoint
SHA-256, epoch, device, case count, residue count, and output schema.

The complete chain can also be run from the repository root with:

```bash
python scripts/run_1crn_example.py
```
