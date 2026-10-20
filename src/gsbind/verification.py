import hashlib
import json

import numpy as np
import pandas as pd


def verify_hashes(root):
    inventory = pd.read_csv(root / 'FILE_INVENTORY.csv')
    for row in inventory.itertuples():
        path = root / row.path
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8*1024*1024), b''):
                digest.update(block)
        if path.stat().st_size != row.bytes or digest.hexdigest() != row.sha256:
            raise ValueError('File content mismatch: ' + row.path)
    return dict(status='passed', sha256_verified_files=len(inventory))


def verify(root):
    records = []
    expected = {'main/train.csv': 20801, 'main/holo_test.csv': 988, 'main/apo_af2_test.csv': 874,
                'allosteric/input_manifest.csv': 504, 'ssm220/manifest.csv': 220, 'functional/manifest.csv': 3}
    missing = []
    for name, count in expected.items():
        frame = pd.read_csv(root / 'datasets' / name, keep_default_na=False)
        if len(frame) != count:
            raise ValueError(f'{name}: expected {count}, found {len(frame)}')
        for column in ['compact_path', 'receptor_pdb_path']:
            for value in frame[column]:
                if not value or not (root / value).is_file():
                    missing.append(dict(manifest=name, column=column, path=value))
        records.append(dict(manifest=name, cases=len(frame)))
    for name in ['functional']:
        frame = pd.read_parquet(root / f'datasets/{name}/' / ('residues.parquet' if name == 'allosteric' else 'complete_features.parquet'))
        hidden = np.load(root / f'datasets/{name}/hidden.npy', mmap_mode='r')
        if hidden.shape != (len(frame), 128):
            raise ValueError('Hidden row alignment failure: ' + name)
    frame = pd.read_parquet(root/'datasets/allosteric/complete_features.parquet')
    hidden = np.load(root/'datasets/allosteric/hidden_centered_conditions.npy',mmap_mode='r')
    assert hidden.shape==(7,len(frame),128)
    np.testing.assert_array_equal(hidden[0],hidden[1])
    contract = json.loads((root/'datasets/allosteric/FOUR_METHOD_CONTRACT.json').read_text())
    assert contract['methods']==['geometry','amino','hidden','full']
    for identifier in pd.read_csv(root/'datasets/allosteric/input_manifest.csv').uniprot:
        completion = json.loads((root/'datasets/allosteric/features'/identifier/'complete.json').read_text())
        assert completion['origin']=='full_E11_new_forward' and completion['identity_exact']
    for task,count in [('AlloBench',383),('AM',127),('ASBench',168),('PASSer_ASD_source',288),('CASBench',70),('ASD_AlloPPI',21)]:
        assert pd.read_parquet(root/'datasets/allosteric/tasks'/(task+'.parquet')).uniprot.nunique()==count
    checkpoint = root / 'checkpoints/direct128_e11.pt'
    result = dict(status='passed' if not missing else 'failed', manifests=records, missing=missing,
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  symlinks=[str(path.relative_to(root)) for path in root.rglob('*') if path.is_symlink()])
    if missing or result['symlinks']:
        raise ValueError(json.dumps(result))
    (root / 'provenance/verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
