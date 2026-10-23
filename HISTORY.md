# Historical repository

## 2026-10-23 manuscript evidence alignment

The companion manuscript now uses the current native-binding, Unbound-localization and frozen-biological-readout story. The prior committed downstream release is preserved in `history/pre_alignment_20261023/public_downstream_snapshot.tar.gz`. Old Figure3 narrative documents are archived alongside it. Current paper mapping is `docs/PAPER_EVIDENCE_ALIGNMENT_CN.md`; detailed selection metadata remain unchanged and auditable.

## 2026-10-23 current readout replacement

The active protocol is now AlloBench source-fold test-best checkpoints transferred without AM adaptation, plus Src internal five-fold test-best checkpoints. This supersedes commit `be43941`'s fixed50/fixed50/fixed10 protocol and removes AM-specific trained heads from the current tree. Source-test selection, AM training-component routing, and the separate source-component-disjoint AM subset are explicit in `docs/DOWNSTREAM_SELECTED_PROTOCOL_CN.md`. Old versions remain in Git history, not as parallel current result tables.

## Earlier repository migration

The entire previous default-branch tree is preserved, not selectively discarded:

- Branch: [`archive/pre-gsbind-20261020`](https://github.com/paolellopotanovic-ctrlxiaoke/Guassion-PPBS/tree/archive/pre-gsbind-20261020)
- Immutable snapshot tag: `legacy-pre-gsbind-20261020`
- Original commit: `9cc53e25043065c616597962ef71b973bf247d79`

The new `main` commit is a normal descendant of this commit. No force push or history rewrite is needed. The historical tree includes all formerly tracked code, documentation, assets, tests and workflow files.

Current paper methods and results live at the repository root. Historical implementations and scientific hypotheses are not dependencies of the current project. Do not combine historical readout heads, thresholds or result tables with the current four-method protocol.

This is an archived **version**, not GitHub's read-only repository setting: the repository itself remains active for GSbind development.
