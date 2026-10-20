import hashlib
import json

import numpy as np
import pandas as pd
import torch

from gaussian_direct.data import collate_single, load_checkpoint_model, load_compact_npz, model_arguments_from_batch
from .surface import SURFACE_CONDITIONS, SurfaceContextCutoff


def run(root, manifest, output, device='cpu', limit=0):
    output.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(manifest, keep_default_na=False).to_dict('records')
    model, payload = load_checkpoint_model(root / 'checkpoints/direct128_e11.pt', device)
    for row in rows[:limit or len(rows)]:
        identity = row['apex_case_id']
        item = load_compact_npz(row['compact_path'], case_id=identity, labels=row.get('labels', ''), label_mask=row.get('label_mask', ''), input_state=row.get('input_state', 'holo'))
        batch = collate_single([item])
        arguments = model_arguments_from_batch(batch, torch.device(device))
        logits, hidden, audits = [], [], []
        for condition in SURFACE_CONDITIONS:
            with torch.inference_mode(), SurfaceContextCutoff(model.gaussian, condition) as controller:
                result = model(**arguments)
                logits.append(result.detach().float().cpu().numpy())
                hidden.append(model.gaussian.last_fused.detach().float().cpu().numpy())
                audits.append(dict(condition=condition, sites=controller.audit))
        np.testing.assert_array_equal(logits[0], logits[1])
        np.testing.assert_array_equal(hidden[0], hidden[1])
        digest = hashlib.sha256(identity.encode()).hexdigest()[:20]
        np.savez_compressed(output / (digest + '.npz'), conditions=np.asarray(SURFACE_CONDITIONS), logits=np.asarray(logits), fused=np.asarray(hidden))
        (output / (digest + '.json')).write_text(json.dumps(dict(case_id=identity, audit=audits), indent=2))
        print(identity, 'complete', flush=True)
