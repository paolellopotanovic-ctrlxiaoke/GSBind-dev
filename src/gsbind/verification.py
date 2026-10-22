import hashlib
import json

import pandas as pd


def verify_hashes(root):
    inventory = pd.read_csv(root / 'FILE_INVENTORY.csv')
    for row in inventory.itertuples():
        path = root / row.path
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if path.stat().st_size != row.bytes or digest.hexdigest() != row.sha256:
            raise ValueError('File content mismatch: ' + row.path)
    return dict(status='passed', sha256_verified_files=len(inventory))


def verify(root):
    from .downstream import TASKS, load_inputs, split_masks, replay
    records = []
    for name, count in [('main/train.csv', 20801), ('main/holo_test.csv', 988),
                        ('main/apo_af2_test.csv', 874), ('ssm220/manifest.csv', 220)]:
        frame = pd.read_csv(root / 'datasets' / name)
        if len(frame) != count:
            raise ValueError('Manifest count mismatch: ' + name)
        records.append(dict(manifest=name, cases=len(frame)))
    for task in TASKS:
        frame, features, matrices = load_inputs(root, task)
        for fold in range(5):
            split_masks(frame, task, fold)
        records.append(dict(task=task, labels=len(frame), feature_rows=len(features)))
    replay(root, root / 'runs/verification/downstream')
    checkpoint = root / 'checkpoints/direct128_e11.pt'
    result = dict(status='passed', records=records,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  public_probe_data_complete=True,
                  full_backbone_raw_inputs='External bundle required; see data_release/README.md')
    destination = root / 'runs/verification/verification.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    return result
