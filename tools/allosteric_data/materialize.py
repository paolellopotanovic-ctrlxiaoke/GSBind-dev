import json

import numpy as np
import pandas as pd

from common import ROOT, SUPPLIED, write_json
from gsbind.readouts import centered_hidden


def main():
    manifest = pd.read_csv(ROOT / 'manifest_with_external.csv')
    membership = pd.read_csv(ROOT / 'homology/membership.csv').set_index('uniprot')
    frames, hidden, conditions = [], [], None
    coverage = []
    for row in manifest.itertuples():
        folder = ROOT / 'features' / row.uniprot
        if not (folder / 'complete.json').exists():
            coverage.append(dict(uniprot=row.uniprot, status='no_valid_features'))
            continue
        frame = pd.read_parquet(folder / 'residues.parquet')
        with np.load(folder / 'surface.npz') as arrays:
            if conditions is None:
                conditions = arrays['conditions'].tolist()
            assert conditions == arrays['conditions'].tolist()
            values = arrays['hidden']
            assert values.shape == (7, len(frame), 128)
            assert np.isfinite(values).all()
            np.testing.assert_array_equal(values[0], values[1])
        for column in membership.columns:
            frame[column] = membership.loc[row.uniprot, column]
        frame['matrix_row'] = np.arange(sum(len(item) for item in frames), sum(len(item) for item in frames)+len(frame))
        frames.append(frame)
        hidden.append(values)
        coverage.append(dict(uniprot=row.uniprot, status='included', residues=len(frame)))
    frame = pd.concat(frames, ignore_index=True)
    hidden = np.concatenate(hidden, axis=1)
    center = np.asarray([centered_hidden(frame, values) for values in hidden], dtype=np.float32)
    np.save(ROOT / 'hidden_centered_conditions.npy', center)
    frame.to_parquet(ROOT / 'complete_features.parquet', index=False)
    write_json(ROOT / 'conditions.json', conditions)
    pd.DataFrame(coverage).to_csv(ROOT / 'feature_coverage_final.csv', index=False)
    tasks = ROOT / 'tasks'
    tasks.mkdir(exist_ok=True)
    positive = pd.read_csv(ROOT / 'allobench_positive_positions.csv')
    positions = set(zip(positive.uniprot, positive.residue_index_0))
    allo = frame[frame.uniprot.isin(positive.uniprot)].copy()
    allo['label'] = [int((row.uniprot, row.residue_index_0) in positions) for row in allo.itertuples()]
    allo.to_parquet(tasks / 'AlloBench.parquet', index=False)
    am = pd.read_csv(SUPPLIED / 'AM/AM_processed.csv', sep='\t')
    am['position'] = am.mut_name.str.extract(r'_\w(\d+)\w$').astype(int)
    am['amino'] = am.mut_name.str.extract(r'_(\w)\d+\w$')
    am['excluded_untraced_source'] = am.mut_name.eq('P62593_M1N')
    labelled = []
    for (identifier, position), group in am.groupby(['uniprot_id', 'position']):
        valid = frame[frame.uniprot.eq(identifier) & frame.residue_index_0.eq(position-1)]
        reason = 'included'
        if group.excluded_untraced_source.any():
            reason = 'untraced_P62593_M1N'
        elif group.label.nunique() != 1:
            reason = 'conflicting_substitution_labels_at_WT_position'
        elif len(valid) != 1:
            reason = 'missing_features_or_position'
        elif valid.amino_acid.iloc[0] != group.amino.iloc[0]:
            reason = 'WT_identity_mismatch'
        labelled.append(dict(uniprot=identifier, residue_index_0=position-1, label=int(group.label.max()), status=reason, variants=len(group)))
    labelled = pd.DataFrame(labelled)
    labelled.to_csv(ROOT / 'am_position_audit.csv', index=False)
    keep = labelled[labelled.status.eq('included')][['uniprot', 'residue_index_0', 'label']]
    mutation = frame.merge(keep, on=['uniprot', 'residue_index_0'], validate='one_to_one')
    mutation.to_parquet(tasks / 'AM.parquet', index=False)
    external = pd.read_csv(ROOT / 'external_positive_positions.csv')
    summary = {}
    for dataset, group in external.groupby('dataset'):
        pairs = set(zip(group.uniprot, group.residue_index_0))
        current = frame[frame.uniprot.isin(group.uniprot)].copy()
        current['label'] = [int((row.uniprot, row.residue_index_0) in pairs) for row in current.itertuples()]
        current.to_parquet(tasks / (dataset+'.parquet'), index=False)
        summary[dataset] = dict(proteins=int(current.uniprot.nunique()), positions=len(current), positives=int(current.label.sum()))
    summary.update(AlloBench=dict(proteins=int(allo.uniprot.nunique()), positions=len(allo), positives=int(allo.label.sum()), components=int(allo.component.nunique())), AM=dict(proteins=int(mutation.uniprot.nunique()), positions=len(mutation), positives=int(mutation.label.sum()), components=int(mutation.component.nunique())))
    write_json(ROOT / 'task_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
