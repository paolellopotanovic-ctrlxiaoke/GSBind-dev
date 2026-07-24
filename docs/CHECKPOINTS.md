# Checkpoints and release assets

The repository must not commit `.pt`, `.pth`, or `.ckpt` files to normal Git
history. Model weights belong in an immutable GitHub Release or Zenodo record.

## Primary inference checkpoint

- Registry name: `m4_holo20k_v9_clean_epoch4`
- Filename: `m4_holo20k_v9_clean_epoch4.pt`
- Size: 46,437,755 bytes
- SHA-256: `d6b418e21d8fc43f347f7c4438e2339877dd3afb240c6196aa7eff896caaee14`
- Epoch/global step: 4/10,000
- Parameters/state tensors: 4,360,516/791
- Required loading mode: strict, zero missing or unexpected keys
- Immutable asset URL:
  `https://github.com/paolellopotanovic-ctrlxiaoke/ckpt1/releases/download/v0.1.0/m4_holo20k_v9_clean_epoch4.pt`

## Publication verification

1. Download into a clean directory with `scripts/download_checkpoint.py`.
2. Run `scripts/validate_checkpoint.py --registry-name ...`.
3. Require the registered size and SHA-256 to match before loading.
4. Run the CUDA inference and one-step training equivalence gates before using
   the release to claim exact end-to-end scientific reproduction.

Uploading a renamed, re-saved, averaged, or optimizer-stripped checkpoint would
change the checksum and is not an equivalent asset. Additional checkpoints used
in paper tables should each receive their own registry record and immutable
hash; do not overwrite the primary name.
