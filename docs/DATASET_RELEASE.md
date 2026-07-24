# Holo20k v9 dataset release

## Immutable identity

The paper training manifest is published as the GitHub Release asset
`holo20k_gpsfun_clusterdiverse_v9_mmseqclean_train.csv`.

| Field | Value |
|---|---:|
| Bytes | 46,063,708 |
| SHA-256 | `65b1df43dd4e00420dea8f59c8bbdc03d5a8fb6b33ed511fc469768ec0799d0d` |
| Rows / unique case IDs / unique PDBs | 20,000 / 20,000 / 20,000 |
| Clusters / maximum cluster occupancy | 9,812 / 20 |
| Positive residues | 607,742 |
| All encoded residues | 4,180,585 |
| Valid masked residues | 4,174,326 |
| Positive rate over all / valid residues | 14.5372% / 14.5590% |

The original CSV is retained byte-for-byte even though
`receptor_pdb_path` records the historical training-server path. Run
`scripts/prepare_holo20k.py` to create a runnable local copy. The script changes
only this path field and records both hashes.

## Structure source and registered repair

Each `holo_pdb` is fetched from the public Pinder 2024-02 `pdbs` bucket. One
source structure, `6z3m__S1_Q6NW40-L.pdb`, contains 14 four-digit B-factors that
overflow the fixed-width PDB field. The preparation script refuses to modify it
unless its source SHA-256 is
`be54687c9aa1f535e2c3779c95480b98271e0a56a360c1de8ffeed7af4f9dd33`, clips
only those B-factors to `999.99`, and requires the resulting SHA-256
`9e2e9b3d8be4e9f73f695c2edf53f12def562bc8f4f650077af284a5125b74be`.
Coordinates, atoms, residues, labels, and manifest ordering are unchanged.

## Audit bundle

`holo20k_v9_audit_bundle.tar.gz` contains:

- the v8-to-v9 construction summary;
- all 36 distribution-matched replacement pairs;
- the two residue-ordinal repair records;
- final and reserve MMseq30/cov80 summaries and zero-hit tables;
- the exact final and reserve FASTA query sets;
- the independently rerun release distribution audit;
- the strict four-cache audit; and
- the registered PDB fixed-width repair record.

The final MMseq2 audit records the exact target FASTA SHA-256 and MMseqs2
revision. External test structures and sequences are not duplicated in this
repository; rebuild the target FASTA from the registered GPSFun, Pinder-S
apo/AF2, and complete Pinder-Large apo sources before rerunning the command.

## Storage

The validated cache build occupied approximately:

| Cache | Size |
|---|---:|
| Tensor | 86 GB |
| Residue graph | 3.5 GB |
| Surface | 5.7 GB |
| Static geometry | 9.4 GB |

Allow at least 130 GB for structures, temporary files, caches, checkpoints, and
logs. Cache files are deterministic build products and are not Release assets.
