"""Publish an additive six-panel figure, source tables and claim alignment."""

import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figure3_gaussian_surface_20261021'
REPOSITORY=ROOT.parent/'GSBind_source_data'
RELEASE='fig3_gaussian_surface_v20261021'
PANELS={
    'A_Surface_dependence':['intervention_case_metrics.csv','intervention_system_metrics.csv','intervention_summary.csv','intervention_roster.csv'],
    'B_Dose_response':[],
    'C_Local_pair_priority':['target_minus_random_case_metrics.csv','target_minus_random_summary.csv','local_intervention_mask_audit.csv'],
    'D_Conformational_transfer':['pair_model_metrics.csv','pair_mapping_audit.csv','A_pair_and_H_packing_summary.csv','A_pair_and_H_packing_paired_contrasts.csv','sensitivity_all_pair_summary.csv','sensitivity_cluster_pair_summary.csv'],
    'E_Correct_candidate_retention':['stability_case_metrics.csv','C_stability_summary.csv','C_stability_paired_contrasts.csv'],
    'F_Spatial_example':['F_all_example_candidates.csv','F_example_metrics.csv','F_example_residues.csv','F_example_geometry.npz','F_EXAMPLE_SELECTION.json','F_injection_audit.json']}
TOOLS=['build_figure3_gaussian_surface.py','plot_figure3_gaussian_surface.py','validate_figure3_gaussian_surface.py','package_figure3_gaussian_surface.py']
DOCS=['FIGURE3_GAUSSIAN_SURFACE_PROTOCOL_CN_20261021.md','FIGURE3_GAUSSIAN_SURFACE_RESULTS_CN_20261021.md']


def estimate(summary,state,condition,metric):
    return summary[summary.input_state.eq(state)&summary.condition.eq(condition)&summary.metric.eq(metric)].iloc[0]


def numerical_claims():
    summary=pd.read_csv(OUT/'intervention_summary.csv')
    contrasts=pd.read_csv(OUT/'target_minus_random_summary.csv')
    descriptions={
        'A':'Gaussian surface feedback is used for accurate binding-site prediction.',
        'B':'Partial attenuation is tolerated better than full removal; surface and pair interventions have distinct graded responses.',
        'C':'Pair removal near native high-score candidates causes greater ranking damage and correct-candidate loss than same-count random removal.',
        'D':'Higher absolute binding accuracy is retained on matched unbound inputs; absolute accuracy is distinct from the amount of AP decline.',
        'E':'Cross-state candidate overlap must be interpreted jointly with retained true interface candidates.',
        'F':'A disclosed positive illustrative case spatially connects pathway removal to interface priority changes.'}
    values={letter:[] for letter in descriptions}
    for state in ['holo','apo','af2']:
        value=estimate(summary,state,'surface_zero','ap_loss')
        values['A'].append(f'{state}: surface AP loss {100*value.estimate:.3f} pp [{100*value.lower:.3f}, {100*value.upper:.3f}], {int(value.systems)} systems')
        half=estimate(summary,state,'surface_half','ap_loss')
        pair=estimate(summary,state,'P_zero','ap_loss')
        values['B'].append(f'{state}: half-surface AP loss {100*half.estimate:.3f} pp; pair-zero loss {100*pair.estimate:.3f} pp')
        for metric in ['ap_loss','correct_lost_per_budget']:
            value=contrasts[contrasts.input_state.eq(state)&contrasts.metric.eq(metric)].iloc[0]
            values['C'].append(f'{state}: extra {metric} {100*value.estimate:.3f} pp [{100*value.lower:.3f}, {100*value.upper:.3f}]')
    paired=pd.read_csv(OUT/'A_pair_and_H_packing_paired_contrasts.csv')
    retained=pd.read_csv(OUT/'C_stability_paired_contrasts.csv')
    for state in ['apo','af2']:
        value=paired[paired.input_state.eq(state)&paired.baseline.eq('pesto_official')&paired.metric.eq('state_ap')].iloc[0]
        values['D'].append(f'{state}: GSBind-PeSTo AP {value.estimate:.5f} [{value.lower:.5f}, {value.upper:.5f}]')
        value=retained[retained.input_state.eq(state)&retained.baseline.eq('pesto_official')&retained.requested_k.eq(0)&retained.metric.eq('retained_correct_per_budget')].iloc[0]
        values['E'].append(f'{state}: GSBind-PeSTo retained correct/budget {100*value.estimate:.3f} pp [{100*value.lower:.3f}, {100*value.upper:.3f}]')
    example=pd.read_csv(OUT/'F_example_metrics.csv').set_index('condition')
    selection=json.loads((OUT/'F_EXAMPLE_SELECTION.json').read_text())
    values['F'].append(selection['case_id'])
    for condition in ['baseline','P_top','surface_zero']:
        values['F'].append(f'{condition}: AP {example.loc[condition,"ap"]:.5f}, precision(top10%) {example.loc[condition,"top_fraction_precision"]:.5f}')
    comparisons=dict(
        A='Native versus 50% and 0% surface feedback; matched inputs within each state',
        B='Observed 0%, 50%, 100% attenuation of surface feedback or pair correction',
        C='P_top minus within-case mean of three same-count random controls',
        D='Paired holo and apo/AF2 AP; four methods on common positions and fixed holo labels',
        E='Paired top10% Jaccard versus shared reference-positive candidates / budget; four methods',
        F='One selected AF2 input: native versus local P_top and surface_zero')
    rows=[]
    for folder in PANELS:
        letter=folder[0]
        rows.append(dict(panel=letter,data_folder=folder,claim=descriptions[letter],numerical_support='; '.join(values[letter]),
            statistical_unit='single illustrative input; no population inference' if letter=='F' else 'system; directed cases equal within system',
            comparison=comparisons[letter],
            endpoint='binding-site reference labels',
            shared_data='A_Surface_dependence/intervention_summary.csv' if letter=='B' else ''))
    return pd.DataFrame(rows)


def main():
    source=REPOSITORY/'source_data'/RELEASE
    plots=REPOSITORY/('GSBind_'+RELEASE)
    for directory in [source,plots/'src',plots/'fig',source/'provenance',source/'paired_inputs']:
        directory.mkdir(parents=True,exist_ok=True)
    claims=numerical_claims()
    claims.to_csv(OUT/'CLAIM_DATA_ALIGNMENT.csv',index=False)
    claims.to_csv(source/'CLAIM_DATA_ALIGNMENT.csv',index=False)
    index=[]
    for folder,files in PANELS.items():
        target=source/folder
        target.mkdir(exist_ok=True)
        for name in files:
            shutil.copy2(OUT/name,target/name)
            index.append(dict(file=name,relative_path=folder+'/'+name))
        row=claims[claims.panel.eq(folder[0])].iloc[0]
        (target/'DESCRIPTION.md').write_text('# '+folder+'\n\n'+row.claim+'\n\n'+row.numerical_support+'\n\n'+
            ('Uses the shared intervention tables in ../A_Surface_dependence/.\n' if folder.startswith('B') else '')+
            'See ../provenance/FIGURE3_GAUSSIAN_SURFACE_PROTOCOL_CN_20261021.md for estimands, encodings and intervention definitions.\n')
    pd.DataFrame(index).to_csv(source/'DATA_FILE_INDEX.csv',index=False)
    definitions=dict(case_id='Unique directed protein input',system_id='Primary bootstrap unit: interaction system',
        input_state='holo, apo or af2',condition='Intervention identifier, defined in CONDITION_DICTIONARY.csv',
        panel='Original official benchmark panel identifier',cluster_id='Original sequence-cluster identifier; blank means unavailable in this manifest',
        receptor_side='Directed receptor side L or R in the interaction system; not a PDB chain identifier',
        receptor_uniprot='Original receptor UniProt annotation',holo_case_id='Matched directed holo input identifier',
        row_index='Zero-based row in the original frozen manifest',compact_sha256='SHA256 of the original compact inference input',
        num_residues='Full original input residue count',valid_positions='Number of valid labeled residues used for AP and top10% evaluation',
        positives='Number of true binding positives among valid/common positions',positive_fraction='Reference positives divided by valid/common positions',
        common_positions='Number of identity-matched common valid positions',reference_positives='Number of holo-reference positives on common positions',
        state_length='Number of residues in original unbound input',holo_length='Number of residues in matched holo input',
        state_coverage='Identity-matched aligned positions / original state length',holo_coverage='Identity-matched aligned positions / original holo length',
        aligned_pairs='Aligned nongap residue pairs before identity filtering',optimal_alignments='Number of equally optimal alignments; -1 encodes overflow',
        mapping_unique='The fixed sequence aligner returned exactly one optimal alignment',included='Included in the unique-mapping main analysis',
        exclusion_reason='Recorded reason for exclusion; no outcome-based exclusions',
        state_pdb_sha256='SHA256 of the actual state structure used for mapping',holo_pdb_sha256='SHA256 of the actual holo structure used for mapping',
        mean_ca_shift='Mean CA displacement after common-position global Kabsch alignment, angstrom',
        interface_ca_shift='Mean fixed-reference-interface CA displacement under that same global alignment, angstrom',
        ca_rmsd='CA root mean square displacement across common positions after global alignment, angstrom',
        interface_rmsd='CA root mean square displacement at reference-interface positions under global alignment, angstrom',
        native_state_label_disagreements='Number of common positions where original state label differs from fixed holo-reference label',
        native_state_ap='AP using original state labels rather than the fixed common holo reference; secondary endpoint',
        score_spearman='Spearman correlation of paired holo/state model scores across common positions',
        packing_conditional_auc='Positive-negative AUROC using only within-packing-stratum pairs, weighted by comparable pair counts',
        recall50_cost='Tie-inclusive inspected fraction to reach 50% of fixed reference positives',
        recall80_cost='Tie-inclusive inspected fraction to reach 80% of fixed reference positives',
        case_mean='Secondary unweighted directed-case mean; not the main system-equal estimate',
        actual_k='Actual common-position candidate budget, min(K,L) or ceil(0.1L)',
        holo_precision='Reference-positive precision of holo top-budget selection',state_precision='Reference-positive precision of state top-budget selection',
        input_residues='Full input count used to define the original predicted high-score neighborhood',
        target_residues='ceil(10% full input count), selected by baseline score without labels',
        all_three_random_masks_equal_count='All three random masks have exactly the same surface-point count as targeted P_top',
        target_equals_native_top10pct_8A_neighborhood='Independent coordinates reproduce the original native-score 8A surface mask',
        surface_ap_loss='Example-candidate native AP minus surface_zero AP',target_ap_loss='Example-candidate native AP minus P_top AP',
        target_extra_loss='Example-candidate three-random mean AP minus P_top AP',
        ap='Average precision across valid original binding labels for this input',native_ap='Native baseline AP on the same labels/mask',
        ap_loss='Native AP minus intervention AP; positive means damage',
        top_fraction_precision='True binding positives among intervention top ceil(0.1 L) valid residues, divided by that budget',
        correct_retained_per_budget='Native true-positive top10% candidates still selected after intervention, divided by fixed budget',
        correct_lost_per_budget='Native true-positive top10% candidates absent from intervention top10%, divided by fixed budget',
        interface_priority_loss='Mean native-minus-intervention percentile rank among reference positives; high score has high priority',
        background_priority_loss='Same priority loss among reference negatives',
        native_candidate_score_drop='Mean native-minus-intervention probability on fixed native top10% candidates; not ranking damage',
        estimate='System-equal mean; cases equal within each state/system',
        lower='2.5 percentile of 10000 system-bootstrap means',upper='97.5 percentile of 10000 system-bootstrap means',
        systems='Number of supporting systems',cases='Number of unique directed cases',metric='Name of the summarized metric',
        model='Frozen prediction method identifier',requested_k='0 means ceil(10% common valid positions); otherwise fixed K',
        jaccard='Intersection/union of paired-state selected candidate sets',
        retained_correct_per_budget='True reference positions selected in both structures / fixed candidate budget',
        retained_correct_count='Count of shared selected true reference positions',
        holo_ap='AP using holo scores on sequence-audited common positions and fixed holo reference',
        state_ap='AP using unbound scores on exactly the same positions and reference labels',
        ap_change='Unbound AP minus paired holo AP',
        eligible='Post-analysis illustrative-case eligibility; does not filter any population estimate',
        selection_distance='Sum of absolute percentile distances to eligible-group medians; minimum used, not largest damage',
        label='Original reference binding label',label_mask='Original valid-label mask',
        residue_index_0='0-based input-residue order; not PDB or UniProt numbering',
        center_x='Model residue center x in angstrom',center_y='Model residue center y in angstrom',center_z='Model residue center z in angstrom',
        target_surface_fraction='Fraction of native sampled surface points removed in P_top',
        surface_points='Number of actual sampled surface points',target_surface_points='Number of surface points in the fixed native high-score 8A neighborhood')
    dictionary=[]
    for entry in index:
        if not entry['file'].endswith('.csv'):
            continue
        for column in pd.read_csv(source/entry['relative_path'],nrows=0).columns:
            meaning=definitions.get(column)
            if column=='identity' and entry['file']=='pair_mapping_audit.csv':
                meaning='Fraction of amino-acid identities among aligned nongap pairs before retaining identical positions'
            if meaning is None and column.endswith('_target'):
                meaning=definitions.get(column.removesuffix('_target'),'Metric')+'; targeted local P_top condition'
            if meaning is None and column.endswith('_random'):
                meaning=definitions.get(column.removesuffix('_random'),'Metric')+'; mean of three seeds within case'
            if meaning is None and column.endswith('_rank'):
                meaning='Average rank / number of valid residues for '+column.removesuffix('_rank')+'; high score means high priority'
            if meaning is None and column.endswith('_top10pct'):
                meaning='Membership in fixed top10% valid-score selection for '+column.removesuffix('_top10pct')
            if meaning is None and column in ['baseline','identity','P_half','P_zero','M_zero','MP_zero','surface_half','surface_zero','P_top','P_random101','P_random211','P_random307']:
                meaning='For F_example_residues: original saved residue probability; for F_all_example_candidates: case AP under '+column
            dictionary.append(dict(file=entry['relative_path'],column=column,definition=meaning or 'Original paired-input metadata or metric; see paired protocol and source field name'))
    pd.DataFrame(dictionary).to_csv(source/'DATA_DICTIONARY.csv',index=False)
    conditions=[
        ('baseline','No intervention'),('identity','Exact unchanged replay'),
        ('surface_half','Surface-context parts of residue feedback/fusion inputs multiplied by 0.5'),
        ('surface_zero','Same surface-context inputs multiplied by 0'),
        ('P_half','Pair correction multiplied by 0.5 before native LayerNorm; single base retained'),
        ('P_zero','Pair correction removed before native LayerNorm; single base retained'),
        ('M_zero','Single-residue projection context_proj input zeroed'),('MP_zero','M_zero and P_zero together'),
        ('P_top','P_zero only at native predicted top10% residue-center 8A surface neighborhood'),
        ('P_random101/211/307','Same number of surface points as P_top, sampled at fixed case-specific seed'),
        ('P_random_mean','Metric average over the three random seeds within each case; never three independent proteins')]
    pd.DataFrame(conditions,columns=['condition','definition']).to_csv(source/'CONDITION_DICTIONARY.csv',index=False)
    shutil.copytree(OUT/'raw',source/'raw',dirs_exist_ok=True)
    for name in ['pair_model_metrics.csv','pair_mapping_audit.csv','stability_case_metrics.csv',
        'A_pair_and_H_packing_summary.csv','A_pair_and_H_packing_paired_contrasts.csv','C_stability_summary.csv',
        'C_stability_paired_contrasts.csv','sensitivity_all_pair_summary.csv','sensitivity_cluster_pair_summary.csv']:
        shutil.copy2(OUT/name,source/'paired_inputs'/name)
    for name in DOCS:
        shutil.copy2(ROOT/'docs'/name,source/'provenance'/name)
    execution=source/'provenance/original_execution_code'
    execution.mkdir(exist_ok=True)
    frozen=json.loads((OUT/'raw/FROZEN_PROTOCOL.json').read_text())
    execution_checks=[]
    for name in ['run_e11_path_cutoff.py','run_e11_official_mmseq_cutoff.py','run_e11_full_cohort_route_dependence.py']:
        path=ROOT.parent/'M4-Gaussian-PPBS/scripts'/name
        expected=next(value for key,value in frozen['sources'].items() if Path(key).name==name)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual==expected
        execution_checks.append(dict(file=name,sha256=actual,matches_original_frozen_execution=True))
        shutil.copy2(path,execution/name)
    pd.DataFrame(execution_checks).to_csv(source/'provenance/ORIGINAL_EXECUTION_HASH_CHECKS.csv',index=False)
    paired_protocol=ROOT/'docs/FIGURE3_BINDING_RELIABILITY_PROTOCOL_CN_20261021.md'
    shutil.copy2(paired_protocol,source/'provenance'/paired_protocol.name)
    for name in ['VALIDATION.json','INDEPENDENT_FIGURE_VALIDATION.json','PORTABLE_REPLAY_VALIDATION.json','SOURCE_HASHES.csv','TEST_RESULTS.txt']:
        shutil.copy2(OUT/name,source/'provenance'/name)
    version={name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy','scikit-learn','pyarrow','matplotlib']}
    (source/'provenance/SOFTWARE_VERSIONS.json').write_text(json.dumps(version,indent=2)+'\n')
    for name in TOOLS:
        shutil.copy2(ROOT/'tools'/name,plots/'src'/name)
    wrapper='''arguments <- commandArgs(trailingOnly = FALSE)
file_argument <- grep("^--file=", arguments, value = TRUE)
script_directory <- dirname(normalizePath(sub("^--file=", "", file_argument[[1]])))
repository <- normalizePath(file.path(script_directory, "../.."))
python <- Sys.getenv("GSBIND_PYTHON", "python")
status <- system2(python, c(shQuote(file.path(script_directory, "plot_figure3_gaussian_surface.py")),
  "--data-dir", shQuote(file.path(repository, "source_data/fig3_gaussian_surface_v20261021")),
  "--figure-dir", shQuote(file.path(script_directory, "../fig"))))
if (status != 0) stop("Figure 3 plotting failed; set GSBIND_PYTHON to the scientific Python interpreter")
'''
    (plots/'src/plot_all.R').write_text(wrapper)
    (plots/'subplot_captions.txt').write_text('\n\n'.join(row.panel+'. '+row.claim+' '+row.numerical_support for row in claims.itertuples())+
        '\n\nSee source_data/'+RELEASE+'/provenance/ for full caption definitions, cohort counts, uncertainty and example selection.\n')
    for path in (OUT/'figures').glob('*'):
        if path.suffix in ['.pdf','.png']:
            shutil.copy2(path,plots/'fig'/path.name)
    (source/'README.md').write_text('''# Gaussian surface pathway Figure 3 (parallel six-panel revision)

This release explains how Gaussian surface modeling supports effective binding prediction. It does not overwrite the original Figure 3 or the earlier binding-reliability revision.

Panels A–C: full completed 1862-input, 12-condition surface/pair interventions. Panels D–E: 788 sequence-audited conformational pairs. Panel F: one explicitly selected positive illustrative AF2 example, not independent validation.

Panel tables are organized in A–F directories. B shares A's intervention tables rather than duplicating numbers. DATA_FILE_INDEX.csv maps every plotting input. CLAIM_DATA_ALIGNMENT.csv, CONDITION_DICTIONARY.csv and DATA_DICTIONARY.csv define the claims, interventions and columns. raw/ contains complete saved residue probabilities and original official tables; paired_inputs/ supports independent metric rebuilding. provenance/ contains protocols, numerical report, validation and test records.

From the source repository root, replot in its established src/fig layout:

```sh
GSBIND_PYTHON=/path/to/scientific/python Rscript GSBind_fig3_gaussian_surface_v20261021/src/plot_all.R
```

Required Python packages: numpy, pandas, scipy, scikit-learn, pyarrow, matplotlib. Tested versions are recorded in provenance/SOFTWARE_VERSIONS.json. No model training or GPU is required to rebuild these metrics and figures from saved results.

To recompute all new metrics, invoke src/build_figure3_gaussian_surface.py with --source-dir pointing to this release's raw/, --paired-dir to paired_inputs/, and --output-dir to a new directory. The geometry validator additionally needs the original full per-case inference cache; this release includes the full selected example and all 1862 local-mask audit rows, not the multi-gigabyte full hidden tensors of every case.
''')
    provenance=[]
    for path in list((plots/'src').glob('*'))+list((source/'provenance').rglob('*')):
        if path.is_file():
            provenance.append(dict(path=str(path.relative_to(REPOSITORY)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    pd.DataFrame(provenance).to_csv(source/'RELEASE_CODE_AND_DOC_HASHES.csv',index=False)
    paper=ROOT.parent/'GSBind_RECOMB27/analysis/figure3_gaussian_surface_20261021'
    paper.mkdir(exist_ok=True)
    for name in DOCS:
        shutil.copy2(ROOT/'docs'/name,paper/name)
    for name in ['VALIDATION.json','INDEPENDENT_FIGURE_VALIDATION.json','PORTABLE_REPLAY_VALIDATION.json','CLAIM_DATA_ALIGNMENT.csv','TEST_RESULTS.txt']:
        shutil.copy2(OUT/name,paper/name)
    for path in (OUT/'figures').glob('GSBind_figure3*'):
        shutil.copy2(path,paper/path.name)
    print(json.dumps(dict(source_data=str(source),plots=str(plots),paper=str(paper),panel_tables=len(index)),indent=2))


if __name__=='__main__':
    main()
