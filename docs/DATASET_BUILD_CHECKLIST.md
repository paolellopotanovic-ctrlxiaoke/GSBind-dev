# Dataset build checklist

This checklist separates a scientifically equivalent rebuild from the exact
paper dataset. A rebuild follows the same rules; exact reproduction additionally
requires the registered final manifest and audit bundle to match byte-for-byte.

## Fast path: exact training manifest

Use this route to reproduce the registered Holo20k v9 training set:

```bash
python scripts/download_dataset.py holo20k_v9_mmseqclean --include-audit
python scripts/prepare_holo20k.py \
  --manifest data/manifests/holo20k_gpsfun_clusterdiverse_v9_mmseqclean_train.csv \
  --pdb-dir data/pinder/2024-02/pdbs \
  --output-manifest data/manifests/holo20k_v9.local.csv \
  --report data/manifests/holo20k_v9.prepare.json --workers 32
```

The downloaded CSV must retain SHA-256
`65b1df43dd4e00420dea8f59c8bbdc03d5a8fb6b33ed511fc469768ec0799d0d`.
The localized CSV intentionally has a different hash because its
`receptor_pdb_path` values point to the user's machine. The preparation report
binds both hashes. All biological labels and identifiers remain unchanged.

Continue at step 7 to build caches. Steps 1-6 below reconstruct and audit the
dataset from raw Pinder inputs; they are not required when using the immutable
paper manifest.

## Inputs

- Pinder 2024-02 `index.parquet`, receptor/partner holo PDBs, native dimers, and
  residue mapping parquet files.
- GPSFun test manifest or FASTA.
- Pinder-S apo/AF2 manifest.
- complete Pinder-Large apo manifest.
- MMseqs2 on `PATH`.

## 1. Build per-partner holo contact events

```bash
python scripts/dataset/build_pinder_manifest_from_index.py \
  --index-path /data/pinder/2024-02/index.parquet \
  --pinder-root /data/pinder/2024-02 \
  --output-dir work/holo_events --split train --states holo \
  --contact-cutoff 5.0 --min-mapping-coverage 0.70 \
  --target-cases 0 --max-residues 0
```

The builder downloads only required Pinder assets, maps heavy-atom contacts to
strict receptor residue ordinals, and records failures. Review
`apex_paired_manifest_summary.json`; do not silently discard mapping errors.

## 2. Union all observed partners

```bash
python scripts/dataset/union_partner_labels.py \
  --input work/holo_events/apex_paired_manifest.csv \
  --output work/holo_union.csv --summary work/holo_union_summary.json
```

The union is allowed only when structure identity and full residue order match.
The result retains partner count, system IDs, and source structures.

## 3. Build the deterministic Pinder-train candidate table

```bash
python scripts/dataset/build_pinder_holo_candidates.py \
  --index /data/pinder/2024-02/index.parquet \
  --output work/candidates.csv --summary work/candidates_summary.json \
  --cluster-cap 20 --target 100000 \
  --pinder-root /data/pinder/2024-02 --require-available
```

Join candidate provenance to `holo_union.csv` by normalized holo structure
basename. Reject missing or ambiguous joins. Selection is deterministic and
never increases the cluster cap when a target cannot be reached.

## 4. Run the mandatory leakage audit

```bash
python scripts/dataset/mmseqs_leakage_audit.py \
  --train-manifest work/selected.csv \
  --test-source GPSFun=/data/tests/gpsfun.csv \
  --test-source PinderS_apo_AF2=/data/tests/pinder_s_apo_af2.csv \
  --test-source PinderLarge_apo=/data/tests/pinder_large_apo.csv \
  --structure-root /data/pinder/2024-02 \
  --work-dir work/mmseqs --output-hits work/mmseqs/hits.csv \
  --output-summary work/mmseqs/summary.json --threads 32
```

This command is fixed at `--min-seq-id 0.30 -c 0.80 --cov-mode 0`. Missing
sequences or any hit fail the command. Add official validation/test sources if
they participate in selection or reporting.

## 5. Reconstruct the registered v9 replacement

The paper v9 starts from v8, removes 36 leaking structures, inserts 36
independently clean reserve structures in the original row positions, and
repairs two residue-ordinal mappings without changing positive labels.

```bash
python scripts/dataset/build_holo20k_v9.py \
  --base-manifest artifacts/holo20k_v8.csv \
  --leak-hits artifacts/v8_leaking_structures.csv \
  --reserve-manifest artifacts/v9_clean_reserve.csv \
  --reserve-audit-summary artifacts/v9_clean_reserve_mmseq.json \
  --output work/holo20k_v9.csv \
  --replacement-log work/v9_replacements.csv \
  --summary work/v9_rebuild_summary.json \
  --cluster-cap 20 --expected-replacements 36
```

Apply the published ordinal-repair table without changing labels:

```bash
python scripts/dataset/apply_ordinal_repairs.py \
  --manifest work/holo20k_v9.csv \
  --repairs artifacts/v9_ordinal_repairs.json \
  --output work/holo20k_v9_repaired.csv \
  --audit work/v9_ordinal_repair_audit.json --expected-repairs 2
```

The repair table is part of the dataset release asset and pins the old ordinal
string checksum before applying each replacement. Rerun strict ordinal
alignment and require final SHA-256
`65b1df43dd4e00420dea8f59c8bbdc03d5a8fb6b33ed511fc469768ec0799d0d`.
The final artifact bundle, not aggregate statistics alone, defines exact v9.

## 6. Audit distribution and manifest integrity

```bash
python scripts/dataset/audit_distribution.py \
  --manifest work/holo20k_v9.csv --output work/distribution.json \
  --expected-rows 20000 --max-cluster-cap 20 --require-unique-paths
python main.py audit-manifest --manifest work/holo20k_v9.csv
```

Expected registered values include 20,000 rows and unique paths, 9,812 clusters,
607,742 positive residues, 4,180,585 total residues, and global positive rate
0.1453724777752396.

## 7. Build and audit all caches

```bash
python main.py build-tensor-cache --manifest data/manifests/holo20k_v9.local.csv \
  --cache-dir cache/tensor --workers 32
python main.py build-graph-cache --manifest data/manifests/holo20k_v9.local.csv \
  --tensor-cache-dir cache/tensor --graph-cache-dir cache/graph \
  --ks 16,24,32 --workers 32
python main.py build-surface-cache --manifest data/manifests/holo20k_v9.local.csv \
  --tensor-cache-dir cache/tensor --residue-graph-cache-dir cache/graph \
  --output-dir cache/surface --states holo --surface-kind shrake \
  --surface-points 1024 --surface-graph-k 16 \
  --surface-cache-version s1_surface_shrake_v1
python main.py build-static-cache --manifest data/manifests/holo20k_v9.local.csv \
  --tensor-cache-dir cache/tensor --residue-graph-cache-dir cache/graph \
  --surface-cache-dir cache/surface --output-dir cache/static \
  --surface-cache-version s1_surface_shrake_v1 \
  --geometry-cache-version e89_dense_geometry_v1 --surface-graph-k 32
python scripts/audit_cache.py --manifest data/manifests/holo20k_v9.local.csv \
  --tensor-cache-dir cache/tensor --graph-cache-dir cache/graph \
  --surface-cache-dir cache/surface --static-cache-dir cache/static \
  --output work/cache_audit.json
```

Only a zero-error cache report permits smoke inference or training.

The validated cache audit requires 20,000 rows and unique paths with
`missing_tensor`, `missing_graph`, `missing_surface`, `missing_static`,
`overlay_mismatch`, `bad_static`, and `zero_valid_rows` all equal to zero.
