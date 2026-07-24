# Data contract

M4 uses one manifest row per receptor structure. Training and labeled evaluation
require true residue labels; `prepare-input` generates technical zero labels only
for label-free prediction cache compatibility.

## Required manifest fields

- `apex_case_id`: non-empty, globally unique case identifier.
- `input_state`: `holo`, `apo`, or `af2`.
- `receptor_pdb_path`: local receptor-only PDB path.
- `input_residue_ids`: semicolon-delimited residue ordinals in strict encoder order.
- `labels`: semicolon-delimited binary labels in the same order.
- `label_mask`: semicolon-delimited validity mask in the same order.
- `num_residues`: exact number of encoded residues.
- `cluster_id`: required for cluster-aware training selection and splitting.

Recommended provenance fields include Pinder system ID, receptor side, UniProt,
source holo structure, union partner count, mapping coverage, label source, and
dataset version. The machine-readable schema is `schemas/manifest.schema.json`.

## Label construction

For each Pinder train complex, define a positive holo receptor residue when any
of its heavy atoms lies within 5 A of any partner heavy atom. Transfer contacts
through Pinder residue-to-UniProt mappings. Never silently truncate, shift, or
align label arrays by position. A receptor observed with multiple partners is
represented by the residue-wise union of all valid partner contact events.

`scripts/dataset/union_partner_labels.py` rejects a group unless every member has
the identical receptor structure key and identical residue ordinal sequence.

## Leakage rule

The publication rule is fixed to MMseqs2 sequence identity at least 0.30 and
coverage at least 0.80 with `--cov-mode 0`. Audit against all reported external
tests:

- GPSFun;
- Pinder-S apo and AF2;
- complete Pinder-Large apo.

Official Pinder validation/test structures should be added whenever they are
used for model selection or reporting. Any hit causes
`mmseqs_leakage_audit.py` to exit nonzero. Missing source structures also abort
the audit.

## Four cache families

Every row must have exactly one compatible tensor, residue graph, SHRake surface,
and static geometry cache. Run `scripts/audit_cache.py` before training. It
requires zero missing entries, zero label-overlay mismatches, zero invalid static
entries, and zero rows without valid residues.

The exact construction sequence is in `DATASET_BUILD_CHECKLIST.md`; registered
paper statistics and hashes are in `assets/datasets.json`.
