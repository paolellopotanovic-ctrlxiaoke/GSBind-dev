# Experiment reproduction

## M4 Holo20k v9 clean

The registered training experiment is from-scratch exact M4 with global routing,
batch size 2, gradient accumulation 4, learning rate `5e-5`, weight decay
`1e-4`, cosine scheduling, warmup ratio 0.05, minimum learning rate `1e-6`, and
20 epochs. The complete architecture and loss flags are pinned in
`configs/m4_holo20k.yaml`.

Before launch, verify:

```bash
python -m pytest -q
python scripts/validate_checkpoint.py CHECKPOINT \
  --registry-name m4_holo20k_v9_clean_epoch4
python scripts/dataset/audit_distribution.py --manifest DATASET.csv \
  --output distribution.json --expected-rows 20000 \
  --max-cluster-cap 20 --require-unique-paths
python scripts/audit_cache.py --manifest DATASET.csv \
  --tensor-cache-dir TENSOR --graph-cache-dir GRAPH \
  --surface-cache-dir SURFACE --static-cache-dir STATIC \
  --output cache_audit.json
```

Then replace only path placeholders in `configs/m4_holo20k.yaml` and run:

```bash
python main.py train --config configs/m4_holo20k.yaml
```

## Evaluation tracks

Use labeled `main.py eval`, never `inference.py`, for benchmark metrics. Report
the same checkpoint on:

- GPSFun;
- Pinder-S apo;
- Pinder-S AF2.

For each track retain manifest hash, successful/failed case IDs, common support,
AUPR, AUROC, case-wise metrics, and raw per-residue predictions. Checkpoint
selection rules must be declared before comparing methods.

## Fine-tuning

`configs/m4_finetune.yaml` is a separate experiment family. Record its parent
checkpoint checksum, filtered training-manifest checksum, leakage audit, reset
or retained optimizer/scheduler state, learning rate, epoch count, and every
changed flag. A fine-tuned result must not be labeled as the from-scratch
Holo20k experiment.

## Exactness

Matching architecture and summary statistics is not exact reproduction. Exact
reproduction requires the registered code commit, checkpoint, final manifest,
audit reports, cache versions, config, seed, and software/GPU environment.
