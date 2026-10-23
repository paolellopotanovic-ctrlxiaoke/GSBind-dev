"""Independently audit plotted values from all saved residue labels/predictions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import auc, precision_recall_curve

from plot_support import METHODS


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'source_data/fig4_downstream_v20261023'


def main():
    checks = []
    metrics = pd.read_csv(DATA / 'shared/METRICS.csv')
    baselines = pd.read_csv(DATA / 'D_pooled_AUPR/random_baselines.csv').set_index('task')
    for task, proteins, count in [('AlloBench', 383, 282), ('AM', 127, 112)]:
        frame = pd.read_csv(DATA / 'shared' / (task+'_predictions.csv.gz'))
        assert frame.uniprot.nunique() == proteins
        assert frame.component.nunique() == count
        assert not frame.duplicated(['case_id', 'residue_index_0']).any()
        baseline = baselines.loc[task]
        assert baseline.positions == len(frame)
        assert baseline.positives == frame.label.sum()
        np.testing.assert_allclose(baseline.prevalence, frame.label.mean())
        folder = 'A_AlloBench' if task == 'AlloBench' else 'B_AM'
        components = pd.read_csv(DATA / folder / 'component_AUPR.csv')
        bars = pd.read_csv(DATA / 'D_pooled_AUPR/bar_values.csv')
        for method in METHODS:
            precision, recall, _ = precision_recall_curve(frame.label, frame[method])
            area = auc(recall, precision)
            np.testing.assert_allclose(area, bars.loc[bars.task.eq(task) & bars.method.eq(method), 'pooled_AUPR'].iloc[0])
            values = []
            for (component, uniprot), group in frame.groupby(['component', 'uniprot']):
                precision, recall, _ = precision_recall_curve(group.label, group[method])
                values.append(dict(component=component, uniprot=uniprot, AUPR=auc(recall, precision)))
            means = pd.DataFrame(values).groupby('component').AUPR.mean().sort_index()
            observed = components.loc[components.method.eq(method)].set_index('component').AUPR.sort_index()
            assert means.index.equals(observed.index)
            np.testing.assert_allclose(means, observed)
            expected = metrics.loc[metrics.task.eq(task) & metrics.scope.eq('all_positions')
                                   & metrics.method.eq(method), 'component_macro_AUPR'].iloc[0]
            np.testing.assert_allclose(means.mean(), expected)
            checks.append(task+'/'+method+': pooled and component AUPR')
    frame = pd.read_csv(DATA / 'shared/Src_predictions.csv.gz')
    constructs = pd.read_csv(DATA / 'C_Src/construct_rho.csv')
    bars = pd.read_csv(DATA / 'C_Src/bar_values.csv').set_index('method')
    assert len(frame) == 538
    assert frame.groupby('physical_position').outer_fold.nunique().eq(1).all()
    variants = pd.read_csv(DATA / 'shared/Src_single_variant_effects.csv')
    variants = variants.loc[variants.wildtype.ne(variants.mutant)].copy()
    variants['absolute_activity_effect'] = variants.activity_effect.abs()
    effects = variants.groupby(['context', 'position']).absolute_activity_effect.mean()
    for case, context in [('src_fl_af2', 'FL'), ('src_kd_af2', 'KD')]:
        group = frame.loc[frame.case_id.eq(case)]
        expected = effects.loc[context].reindex(group.physical_position)
        assert expected.notna().all()
        np.testing.assert_allclose(group.functional_magnitude, expected)
    checks.append('Src: functional magnitudes reconstructed from raw single-variant effects')
    for method in METHODS:
        values = []
        for case, group in frame.groupby('case_id'):
            assert len(group) == 269
            value = spearmanr(group.functional_magnitude, group[method]).statistic
            observed = constructs.loc[constructs.case_id.eq(case) & constructs.method.eq(method), 'rho'].iloc[0]
            np.testing.assert_allclose(value, observed)
            values.append(value)
        np.testing.assert_allclose(np.mean(values), bars.loc[method, 'rho'])
        checks.append('Src/'+method+': complete-OOF and construct mean rho')
    for name, path_column, base in [('FILE_INDEX.csv', 'relative_path', DATA),
                                    ('CODE_HASHES.csv', 'path', ROOT)]:
        for row in pd.read_csv(DATA / name).to_dict('records'):
            path = base / row[path_column]
            assert path.stat().st_size == row['bytes'], str(path)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], str(path)
        checks.append(name+': all sizes and SHA-256')
    report = dict(status='PASS', checks=checks, no_training=True,
                  source_test_selected=True, AM_target_fitting=False,
                  D_estimand='pooled trapezoidal AUPR', case_macro_not_substituted=True)
    (DATA / 'VALIDATION.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
