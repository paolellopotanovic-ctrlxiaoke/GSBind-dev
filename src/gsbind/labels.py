import ast
import json
import numpy as np
import pandas as pd


def rebuild_functional(root, output):
    atlas = pd.read_csv(root / 'datasets/functional/experimental_residues.csv', float_precision='round_trip')
    energy = pd.read_csv(root / 'datasets/functional/raw/inferred_energy_single_variants.csv', float_precision='round_trip')
    folding = energy[energy.assay.eq('folding')][['variant_id', 'position', 'wildtype', 'mutant', 'ddg']].rename(columns={'ddg': 'folding_ddg'})
    binding = energy[~energy.assay.isin(['folding', 'full length RAF1'])].rename(columns={'assay': 'interaction', 'ddg': 'binding_ddg'})
    paired = binding.merge(folding, on=['variant_id', 'position', 'wildtype', 'mutant'], validate='many_to_one')
    records = []
    for (interaction, position), group in paired.groupby(['interaction', 'position']):
        records.append(dict(dataset='KRAS', context='', interaction=interaction, position=position,
                            rebuilt_function=float(group.binding_ddg.abs().mean()), rebuilt_stability=float(group.folding_ddg.abs().mean())))
    src = pd.read_csv(root / 'datasets/functional/raw/energy_single_variants.csv', float_precision='round_trip')
    for (context, position), group in src.groupby(['context', 'position']):
        records.append(dict(dataset='SRC', context=context, interaction='SRC_' + context, position=position,
                            rebuilt_function=float(group.activity_effect.abs().mean()), rebuilt_stability=float(group.folding_effect.abs().mean())))
    rebuilt = pd.DataFrame(records)
    atlas['context'] = atlas.context.fillna('')
    joined = atlas.merge(rebuilt, on=['dataset', 'context', 'interaction', 'position'], validate='many_to_one', how='left')
    if joined.rebuilt_function.isna().any():
        raise ValueError('Missing experimental effect source')
    np.testing.assert_allclose(joined.functional_magnitude, joined.rebuilt_function, atol=1e-12, rtol=0)
    np.testing.assert_allclose(joined.stability_magnitude, joined.rebuilt_stability, atol=1e-12, rtol=0)
    output.mkdir(parents=True, exist_ok=True)
    rebuilt.to_csv(output / 'rebuilt_position_effects.csv', index=False)
    result = dict(status='passed', aligned_experimental_rows=len(joined), source_substitutions_kras=len(paired), source_substitutions_src=len(src),
                  function_max_error=float(abs(joined.functional_magnitude-joined.rebuilt_function).max()),
                  stability_max_error=float(abs(joined.stability_magnitude-joined.rebuilt_stability).max()))
    (root / 'provenance/functional_label_rebuild.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


def rebuild_allosteric(root, output):
    output.mkdir(parents=True, exist_ok=True)
    data = root / 'datasets/allosteric'
    outputs = []
    for task,filename in [('AlloBench','allobench_positive_positions.csv'),('CASBench','casbench_positive_positions.csv')]:
        positive = pd.read_csv(data / filename)
        pairs = set(zip(positive.uniprot,positive.residue_index_0))
        frame = pd.read_parquet(data / 'tasks' / (task+'.parquet'))
        expected = np.asarray([int((row.uniprot,row.residue_index_0) in pairs) for row in frame.itertuples()])
        np.testing.assert_array_equal(frame.label,expected)
        outputs.append(dict(task=task,positions=len(frame),positives=int(expected.sum()),status='passed'))
    positive = pd.read_csv(data / 'external_positive_positions.csv')
    for task,group in positive.groupby('dataset'):
        frame = pd.read_parquet(data / 'tasks' / (task+'.parquet'))
        pairs = set(zip(group.uniprot,group.residue_index_0))
        expected = np.asarray([int((row.uniprot,row.residue_index_0) in pairs) for row in frame.itertuples()])
        np.testing.assert_array_equal(frame.label,expected)
        outputs.append(dict(task=task,positions=len(frame),positives=int(expected.sum()),status='passed'))
    audit = pd.read_csv(data / 'am_position_audit.csv')
    eligible = audit[audit.status.eq('included')][['uniprot','residue_index_0','label']]
    frame = pd.read_parquet(data / 'tasks/AM.parquet')
    joined = frame.merge(eligible,on=['uniprot','residue_index_0'],suffixes=('','_audit'),validate='one_to_one')
    assert len(joined)==len(frame)==len(eligible)
    np.testing.assert_array_equal(joined.label,joined.label_audit)
    outputs.append(dict(task='AM',positions=len(frame),positives=int(frame.label.sum()),status='passed'))
    pd.DataFrame(outputs).to_csv(output / 'label_rebuild.csv',index=False)
    print(json.dumps(outputs,indent=2))


def rebuild_ssm(root, output):
    files = sorted((root / 'datasets/ssm220/raw/assays/validation_output').glob('*_correlation.sc'))
    if len(files) != 14:
        raise ValueError('Expected fourteen source SSM variant tables')
    variants = pd.concat([pd.read_csv(path, sep=r'\s+', float_precision='round_trip') for path in files], ignore_index=True)
    key = ['ssm_parent', 'ssm_seqpos']
    if variants.duplicated(key + ['ssm_letter']).any():
        raise ValueError('Duplicate substitution')
    variants['nonwt'] = variants.ssm_letter.ne(variants.parent_letter)
    bounds = variants.delta_exp_ddg_lb.notna() & variants.delta_exp_ddg_ub.notna() & variants.delta_exp_ddg_lb.le(variants.delta_exp_ddg_ub)
    variants['quality_valid'] = variants.nonwt & ~variants.low_conf & ~variants.avid_doesnt_agree & bounds
    variants['definite_sensitive'] = variants.quality_valid & variants.delta_exp_ddg_lb.ge(2.)
    variants['definite_below'] = variants.quality_valid & variants.delta_exp_ddg_ub.lt(2.)
    grouped = variants.groupby(key, sort=True)
    residues = grouped.agg(n_nonwt_raw=('nonwt', 'sum'), n_quality_valid=('quality_valid', 'sum'),
                           n_definite_sensitive=('definite_sensitive', 'sum'), n_definite_below=('definite_below', 'sum'))
    for name, minimum in [('primary18', 18), ('relaxed1', 1)]:
        eligible = residues.n_quality_valid.ge(minimum)
        label = pd.Series(pd.NA, index=residues.index, dtype='Int64')
        label.loc[eligible & residues.n_definite_sensitive.gt(0)] = 1
        label.loc[eligible & residues.n_definite_below.eq(residues.n_quality_valid)] = 0
        residues[name + '_label'] = label
        residues[name + '_eligible'] = eligible
    residues = residues.reset_index()
    expected = pd.read_csv(root / 'datasets/ssm220/raw/interval_labels.csv').sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(residues.sort_values(key).reset_index(drop=True), expected, check_dtype=False)
    output.mkdir(parents=True, exist_ok=True)
    residues.to_csv(output / 'rebuilt_interval_labels.csv', index=False)
    variants.to_parquet(output / 'source_variants.parquet', index=False)
    legacy = grouped.delta_exp_ddg_center.agg(['max', 'mean', 'size']).reset_index()
    for threshold in [1, 2]:
        legacy[f'sensitive_ge_{threshold}'] = legacy['max'].ge(threshold)
    legacy.to_csv(output / 'rebuilt_legacy_center_labels.csv', index=False)
    print('Rebuilt and verified', len(residues), 'SSM residue labels')
