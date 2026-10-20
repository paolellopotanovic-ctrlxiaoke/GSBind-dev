# Bulk data release

## Current status: public download URL pending

The server has a complete, checksum-verified bundle. This Git repository contains the code, E11 weights, readout weights, data manifests/labels, final CSV metrics, detailed saved predictions and CPU examples. It deliberately does not commit the bulk training structures/caches or large allosteric hidden tensors.

| Artifact | Status |
|---|---|
| `GSbind_paper_20261020.tar.zst` | Exists on the project server; public URL **not yet assigned** |
| Archive bytes | `15323513322` |
| Archive SHA256 | `0bbbc666e9f5ad74ae9dc273642f788c4bf676b3673a2446d0a67c470cfa8c65` |
| Download registration | `bundle.json`, currently `url: null` |
| Files absent from Git | `EXTERNAL_FILES.csv`, exact paths, sizes and SHA256 |
| Full server snapshot | `FULL_BUNDLE_INVENTORY.csv`, 61,801 files; not the public checkout inventory |
| Public checkout inventory | Root `FILE_INVENTORY.csv`; excludes itself and generated `runs/` |

The archive's root directory is `GSbind/`. Its code/docs are the pre-publication snapshot. Install **only its datasets** over the current checkout so publication fixes are retained:

```bash
sha256sum GSbind_paper_20261020.tar.zst
mkdir -p /path/to/temporary_extract
tar --zstd -xf GSbind_paper_20261020.tar.zst -C /path/to/temporary_extract
cp -a /path/to/temporary_extract/GSbind/datasets/. /path/to/Guassion-PPBS/datasets/
cd /path/to/Guassion-PPBS
python tools/check_publication.py --external
python main.py verify
python main.py evidence --task all
```

`check_publication.py --external` requires every registered external file and checks its bytes/hash; it fails rather than silently accepting a partial download. Do not overwrite root `FILE_INVENTORY.csv` with the full-bundle inventory: they describe different packages.

## Which operations work without bulk data?

| Operation | Public clone | Bulk data required |
|---|---|---|
| Read all maintext metrics and detailed saved predictions | Yes | No |
| Recompute metrics from saved predictions: `main.py summarize` | Yes | No |
| CPU example inference, evaluation and Lightning gradient step | Yes | No |
| KRAS/Src frozen readout and raw CSV label reconstruction | Yes | No |
| Main/SSM evidence from saved binding scores | Yes | No |
| Full frozen Allo hidden replay and probe refit | No | Yes |
| All-case surface forward and complete model training | No | Yes |
| Rebuild SSM labels from original large assay tables | No | Yes |

Once a public hosting location is selected, update `bundle.json` and this page with the real URL. Large upstream archives and experimental source files remain subject to their original licensing and attribution.
