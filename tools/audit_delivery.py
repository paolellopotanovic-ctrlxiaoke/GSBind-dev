import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from gaussian_direct.data import load_checkpoint_model
from gsbind.verification import verify


def main():
    os.chdir(ROOT)
    status = verify(ROOT)
    model, payload = load_checkpoint_model(ROOT / 'checkpoints/direct128_e11.pt')
    status['model_parameters'] = sum(parameter.numel() for parameter in model.parameters())
    status['epoch'] = payload['epoch']
    data = pd.read_parquet(ROOT / 'datasets/main/four_model_predictions.parquet')
    keys = ['panel', 'case_id', 'residue_index_0']
    if data.duplicated(keys + ['model']).any():
        raise ValueError('Duplicate four-model prediction')
    common = data.groupby(keys).agg(models=('model', 'nunique'), labels=('label', 'nunique'), masks=('label_mask', 'nunique'))
    assert common.models.eq(4).all() and common.labels.eq(1).all() and common.masks.eq(1).all()
    status['four_model_common_positions'] = len(common)
    base = pd.read_parquet(ROOT / 'datasets/main/surface_predictions.parquet')
    wide = base.pivot(index=keys, columns='condition', values='probability')
    np.testing.assert_array_equal(wide.baseline, wide.identity)
    predictions = data[data.model.eq('gaussian_direct128')].merge(wide.baseline.rename('surface_baseline').reset_index(), on=keys, validate='one_to_one')
    status['main_surface_baseline_max_error'] = float(abs(predictions.score-predictions.surface_baseline).max())
    status['main_baseline_probability_tolerance'] = 2e-7
    status['main_baseline_probability_note'] = 'Saved float32 sigmoid versus canonical sigmoid conversion; compare within 2e-7, not bitwise across numerical backends'
    assert status['main_surface_baseline_max_error'] < status['main_baseline_probability_tolerance']
    status['allosteric_readout'] = json.loads((ROOT / 'results/allosteric/readout_verification.json').read_text())
    status['functional_readout'] = json.loads((ROOT / 'results/functional/readout_verification.json').read_text())
    assert status['allosteric_readout']['methods']==['geometry','amino','hidden','full']
    status['allosteric_protocol'] = 'AlloBench383 and AM127 nested fivefold; four-method fixed transfers'
    status['four_method_refit'] = json.loads((ROOT/'provenance/four_method_refit_verification.json').read_text())
    assert status['four_method_refit']['status']=='passed'
    status['lightning'] = json.loads((ROOT / 'provenance/lightning_smoke.json').read_text())
    status['portable_copy'] = json.loads((ROOT / 'provenance/portable_smoke.json').read_text())
    status['functional_labels'] = json.loads((ROOT / 'provenance/functional_label_rebuild.json').read_text())
    status['structural_covariates'] = json.loads((ROOT / 'provenance/covariate_verification.json').read_text())
    public_audit = json.loads((ROOT / 'provenance/publication_validation.json').read_text())
    status['surface_forward_smoke'] = public_audit['surface_forward_smoke']
    status['unit_tests'] = public_audit['unit_tests']
    status['status'] = 'passed'
    (ROOT / 'provenance/validation_summary.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
