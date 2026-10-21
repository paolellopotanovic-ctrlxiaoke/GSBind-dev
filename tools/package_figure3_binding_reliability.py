"""Validate paired figure metrics and add a parallel source-data release."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import build_figure3_binding_reliability as study

ROOT, OUT = study.ROOT, study.OUT
REPO = ROOT.parent/'GSBind_source_data'
RELEASE = 'fig3_binding_v20261021'


def validate():
    audit = pd.read_csv(OUT/'pair_mapping_audit.csv')
    assert len(audit)==874 and audit.case_id.is_unique
    residues = pd.read_parquet(OUT/'paired_residue_predictions.parquet')
    metrics = pd.read_csv(OUT/'pair_model_metrics.csv')
    assert not residues.duplicated(['case_id', 'state_index']).any()
    assert not residues.duplicated(['case_id', 'holo_index']).any()
    assert set(residues.case_id)==set(audit.loc[audit.included, 'case_id'])
    lookup = metrics.set_index(['case_id', 'model'])
    original = pd.read_parquet(ROOT/'datasets/main/four_model_predictions.parquet')
    originals = {(case, model):group.sort_values('residue_index_0') for (case, model), group in original.groupby(['case_id', 'model'], sort=False)}
    budgets = pd.read_csv(OUT/'fixed_budget_case_metrics.csv').set_index(['case_id', 'model', 'requested_k'])
    stability = pd.read_csv(OUT/'stability_case_metrics.csv').set_index(['case_id', 'model', 'requested_k'])
    errors = []
    for case, group in residues.groupby('case_id', sort=False):
        label = group.reference_label.to_numpy()
        assert len(np.unique(label))==2
        for model in study.MODELS:
            score = group[model+'_state_score'].to_numpy()
            original_state = originals[(case, model)].iloc[group.state_index.to_numpy(int)]
            original_holo = originals[(group.holo_case_id.iloc[0], model)].iloc[group.holo_index.to_numpy(int)]
            np.testing.assert_array_equal(score, original_state.score.to_numpy())
            np.testing.assert_array_equal(group[model+'_holo_score'], original_holo.score.to_numpy())
            np.testing.assert_array_equal(label, original_holo.label.to_numpy())
            replay = average_precision_score(label, score)
            error = abs(replay-lookup.loc[(case, model), 'state_ap'])
            assert error<1e-12
            errors.append(error)
            for requested in study.KS+[0]:
                budget = min(requested, len(group)) if requested else max(1, int(np.ceil(.1*len(group))))
                selected = np.argsort(-score, kind='stable')[:budget]
                assert int(label[selected].sum())==budgets.loc[(case, model, requested), 'hits']
                holo_selected = np.argsort(-group[model+'_holo_score'].to_numpy(), kind='stable')[:budget]
                intersection = set(selected)&set(holo_selected)
                correct = sum(label[index] for index in intersection)
                assert correct==stability.loc[(case, model, requested), 'retained_correct_count']
            for target in [.5, .8]:
                selected = np.argsort(-score, kind='stable')
                required = int(np.ceil(label.sum()*target))
                boundary = np.flatnonzero(np.cumsum(label[selected])>=required)[0]
                cost = np.count_nonzero(score>=score[selected[boundary]])/len(score)
                assert abs(cost-lookup.loc[(case, model), f'recall{int(target*100)}_cost'])<1e-12
    summary = pd.read_csv(OUT/'A_pair_and_H_packing_summary.csv')
    for row in summary.itertuples():
        group = metrics[metrics.input_state.eq(row.input_state)&metrics.model.eq(row.model)]
        assert abs(group.groupby('system_id')[row.metric].mean().mean()-row.estimate)<1e-12
    mapping = audit[audit.included]
    assert mapping.identity.ge(.95).all() and mapping.state_coverage.ge(.8).all() and mapping.holo_coverage.ge(.8).all()
    assert mapping.optimal_alignments.eq(1).all()
    report = dict(starting_cases=len(audit), included_cases=int(audit.included.sum()), excluded_cases=int((~audit.included).sum()),
        included_state_counts=mapping.input_state.value_counts().to_dict(), systems=int(mapping.system_id.nunique()),
        paired_residues=len(residues), maximum_replayed_ap_error=max(errors), replayed_case_model_metrics=len(errors),
        unique_correspondence=True, fixed_reference_labels=True, recall_cost_replay=True, system_estimand_replay=True,
        no_new_model_training=True, all_four_models_same_cases=True)
    report.update(original_score_exact_replay=True, original_reference_label_exact_replay=True, fixed_budget_hits_replay=True,
        cross_state_correct_retention_replay=True)
    (OUT/'VALIDATION.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def release():
    target = REPO/'source_data'/RELEASE
    panels = dict(A='Paired_transfer', B='Interface_deformation', C='Correct_candidate_retention', D='Fixed_budget',
        E='Interface_coverage_cost', F='Spatial_regions', G='AF2_confidence', H='Packing_selectivity', I='Joint_benefit')
    mapping = dict(A=['pair_model_metrics.csv', 'A_H_system_distributions.csv', 'A_pair_and_H_packing_summary.csv', 'A_pair_and_H_packing_paired_contrasts.csv'],
        B=['B_shift_case_metrics.csv', 'B_shift_summary.csv', 'B_shift_paired_contrasts.csv'],
        C=['stability_case_metrics.csv', 'C_system_distributions.csv', 'C_stability_summary.csv', 'C_stability_paired_contrasts.csv'],
        D=['fixed_budget_case_metrics.csv', 'D_system_distributions.csv', 'D_budget_distribution_grid.csv', 'D_budget_summary.csv', 'D_budget_paired_contrasts.csv'],
        E=['A_pair_and_H_packing_summary.csv', 'A_pair_and_H_packing_paired_contrasts.csv',
            'E_full_recall_cost_case_metrics.csv', 'E_full_recall_cost_summary.csv', 'E_full_recall_cost_paired_contrasts.csv'],
        F=['spatial_region_case_metrics.csv', 'F_region_summary.csv', 'F_region_paired_contrasts.csv'],
        G=['G_confidence_candidates.csv', 'G_confidence_summary.csv', 'G_system_distributions.csv', 'G_common_confidence_summary.csv', 'G_common_confidence_paired_contrasts.csv'],
        H=['packing_stratum_case_metrics.csv', 'H_stratum_system_distributions.csv', 'A_pair_and_H_packing_summary.csv', 'A_pair_and_H_packing_paired_contrasts.csv'],
        I=['I_joint_system_metrics.csv', 'I_joint_summary.csv'])
    claims = dict(A='Paired absolute binding accuracy; performance-loss robustness assessed separately.',
        B='Continuous deformation analysis; no claim that every deformation stratum is superior.',
        C='Common true-positive recommendations per fixed top10% budget, not consistency alone.',
        D='Binding precision at a fixed residue budget; all K and baselines retained.',
        E='Retrospective reference-interface coverage cost; lower is better, not a deployable stopping rule.',
        F='Geometric candidate regions; report recall together with actual inspected area, not ligand pockets.',
        G='AF2 candidate confidence and correctness; common groups for direct model comparisons, not calibrated probabilities.',
        H='Binding discrimination within coarse packing strata; not full structural adjustment or Gaussian attribution.',
        I='Joint higher binding AP and lower 80%-reference-recall screening cost within the same system; all systems and comparators retained.')
    numeric = dict(A='GSBind minus PeSTo AP: AF2 +0.0522 [0.0347, 0.0699]; apo +0.0530 [0.0115, 0.0971].',
        B='Fixed <1/1-2/>=2 A strata, all retained. No uniform significant benefit in all strata.',
        C='Correctly retained top10% candidates: AF2 0.5948 vs 0.5233, delta +0.0714 [0.0537, 0.0896]; apo 0.4346 vs 0.3929, delta +0.0417 [0.0069, 0.0767].',
        D='AF2 top10 precision 0.8098 vs 0.7512, delta +0.0586 [0.0326, 0.0853]. Apo top10 delta interval includes zero.',
        E='80% interface recall cost: AF2 0.2709 vs 0.2992, delta -0.0282 [-0.0394, -0.0177]; apo 0.3057 vs 0.3665, delta -0.0608 [-0.1015, -0.0223].',
        F='AF2 top3-region recall 0.3780 vs 0.3381, delta +0.0400 [0.0254, 0.0555]; inspected fractions 0.1369 vs 0.1359. Apo top3 recall delta interval includes zero.',
        G='GSBind precision in common <70/70-90/>=90 mean candidate pLDDT groups: 0.6319/0.7013/0.7370. Descriptive confidence analysis.',
        H='Within-packing AUROC delta: AF2 +0.0194 [0.0111, 0.0282]; apo +0.0451 [0.0168, 0.0747].',
        I='AF2 simultaneous strict wins: PeSTo 198/371 (53.4%), GPSFun 279/371 (75.2%), ScanNet 297/371 (80.1%); apo 22/38 (57.9%), 25/38 (65.8%), 24/38 (63.2%), respectively. System-bootstrap intervals and all quadrants in I_joint_summary.csv.')
    for letter, name in panels.items():
        folder = target/(letter+'_'+name)
        folder.mkdir(parents=True, exist_ok=True)
        for filename in mapping[letter]:
            shutil.copy2(OUT/filename, folder/filename)
        (folder/'DESCRIPTION.md').write_text(f'# Panel {letter}: {name}\n\nSee the frozen protocol and data dictionary. All methods use the same sequence-audited positions and fixed holo reference labels. Point estimates are system-macro; intervals use 10000 system bootstrap draws.\n')
        (folder/'CONCLUSION.md').write_text(claims[letter]+'\n\n'+numeric[letter]+'\n')
    for name in ['raw', 'provenance', 'supplement', 'README']:
        (target/name).mkdir(exist_ok=True)
    for filename in ['paired_residue_predictions.parquet', 'paired_residue_predictions.csv.gz', 'pair_mapping_audit.csv']:
        shutil.copy2(OUT/filename, target/'raw'/filename)
    for path in OUT.glob('sensitivity_*'):
        shutil.copy2(path, target/'supplement'/path.name)
    for path in OUT.glob('I_example_*.csv'):
        shutil.copy2(path, target/'supplement'/path.name)
    shutil.copy2(OUT/'I_joint_validation.json', target/'provenance/I_joint_validation.json')
    for path in (OUT/'figures').glob('I_paired_structure.*'):
        shutil.copy2(path, target/'supplement'/path.name)
    for filename in ['CONTRACT.json', 'VALIDATION.json', 'input_and_code_hashes.csv', 'TEST_RESULTS.txt']:
        shutil.copy2(OUT/filename, target/'provenance'/filename)
    docs = [study.PROTOCOL, ROOT/'docs/FIGURE3_BINDING_RELIABILITY_RESULTS_CN_20261021.md',
        ROOT/'docs/FIGURE3_TASK_STORY_AND_JOINT_BENEFIT_CN_20261021.md',
        ROOT/'docs/FIGURE3_COMMENT_RESPONSE_AND_DENSE_DESIGN_CN_20261021.md']
    for path in docs:
        if path.exists():
            shutil.copy2(path, target/'provenance'/path.name)
    alignment = pd.DataFrame([dict(panel=letter, data_folder=letter+'_'+name, claim=claims[letter], numerical_support=numeric[letter],
        metric_direction='lower is better for coverage cost; higher for accuracy', endpoint='fixed_holo_reference_binding_interface') for letter, name in panels.items()])
    alignment.to_csv(target/'CLAIM_DATA_ALIGNMENT.csv', index=False)
    definitions = dict(case_id='Unique directed state input identifier', holo_case_id='Matched directed holo reference identifier',
        system_id='Protein interaction system; primary bootstrap unit', input_state='apo or af2',
        model='Frozen prediction method identifier; no score modification', estimate='Equal-system mean; in I_joint_summary, proportion of systems in the specified quadrant',
        lower='2.5 percentile of 10000 paired system bootstrap draws', upper='97.5 percentile of 10000 paired system bootstrap draws',
        systems='Number of systems supporting this row', cases='Number of directed cases supporting this row', case_mean='Unweighted directed-case mean, secondary estimand',
        baseline='Comparator; estimate in contrast files is GSBind minus comparator',
        ap_gain='System-mean binding AP of GSBind minus comparator; positive is better',
        cost_saving='System-mean comparator recall80_cost minus GSBind; positive is less screening',
        quadrant='Strict signs: both_better, accuracy_only, cost_only, neither; any exact zero is boundary_tie',
        count='Number of systems in this joint-benefit quadrant',
        directed_cases='Number of directed cases averaged within this system',
        target_recall='Fixed reference-interface recall target; all tenths from 0.1 through 0.9 are retained',
        mean='Equal-system mean precision in the fixed-budget grid',
        q25='25th percentile of system-level precision; not a confidence bound',
        median='Median system-level precision',
        q75='75th percentile of system-level precision; not a confidence bound',
        state_ap='Average precision on aligned state scores and fixed holo reference labels',
        holo_ap='Average precision on aligned holo scores and the same fixed reference labels', ap_change='state AP minus matched holo AP',
        mapping_unique='True only when the fixed sequence aligner returns one optimal alignment',
        retained_correct_per_budget='Both states select a true reference positive, count divided by candidate budget',
        retained_correct_count='Number of true reference positives selected in both states',
        requested_k='5/10/20/40 residues; 0 encodes ceil(10% of common positions)', actual_k='min(requested K,L), or ceil(0.1L) for requested_k=0',
        precision='True reference positives divided by inspected positions', hits='Count of true reference positives inspected',
        recall='Fraction of reference positives inspected', recall50_cost='Inspected residue fraction to 50% reference recall; include boundary ties',
        recall80_cost='Inspected residue fraction to 80% reference recall; include boundary ties',
        inspected_fraction='Fraction of common input residues inspected; region union in F; tie-inclusive score prefix to target_recall in E',
        interface_recall='True reference positives in region union divided by total reference positives',
        requested_regions='Top 1/3/5 geometry-defined regions requested', actual_regions='Number of nonredundant available regions used',
        centers='0-based positions in the ordered common residue frame, not original PDB numbering',
        packing_conditional_auc='AUROC using only within-packing-stratum positive-negative pairs, pair-count weighted',
        mean_ca_shift='Mean CA displacement after all-common-position Kabsch alignment, angstrom',
        interface_ca_shift='Mean CA displacement at fixed reference interface after the same global alignment, angstrom',
        mean_candidate_plddt='AF2 CA B-factor confidence averaged over model top10% candidates',
        confidence_bin='Fixed pLDDT <70,70-90,>=90; common groups use GSBind candidate confidence',
        reference_label='Holo interface-union label projected by verified sequence alignment',
        native_state_label='Original state label preserved for sensitivity analysis',
        state_index='0-based index into original state prediction array', holo_index='0-based index into original holo prediction array',
        stratum='CA8A neighbor-count bins: 0=0-7,1=8-15,2=16-23,3>=24',
        metric='Metric identifier; see frozen protocol and full results report for definition')
    dictionary = []
    for path in sorted(OUT.glob('*.csv')):
        table = pd.read_csv(path, nrows=0)
        dictionary.extend(dict(file=path.name, column=column, definition=definitions.get(column, 'Audit or task-specific field; see frozen protocol and results report')) for column in table.columns)
    pd.DataFrame(dictionary).to_csv(target/'DATA_DICTIONARY.csv', index=False)
    (target/'README/panels.md').write_text('''# Parallel binding-reliability Figure 3

This release does not replace fig3 or fig3_v20261020. Nine panels are supplied with complete source metrics;
a compact six-panel view is also provided. See provenance/ for the frozen protocol and numerical report.
All predictions are unchanged. No allosteric, mutation-effect, or probe-training endpoint is included.
Raw tables retain all four models, mapping failures and prespecified ambiguity/cluster sensitivity analyses.
''')
    plot_folder = REPO/'GSBind_fig3_binding_v20261021'
    for name in ['src', 'fig']:
        (plot_folder/name).mkdir(parents=True, exist_ok=True)
    for path in (OUT/'figures').iterdir():
        shutil.copy2(path, plot_folder/'fig'/path.name)
    code = [Path(study.__file__), ROOT/'tools/render_figure3_paired_example.py', ROOT/'tools/build_figure3_distribution_data.py',
        ROOT/'tools/build_figure3_joint_benefit.py', Path(__file__)]
    for path in code:
        shutil.copy2(path, plot_folder/'src'/path.name)
    plot = (ROOT/'tools/plot_figure3_binding_reliability.R').read_text()
    plot = '''args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep('^--file=', args, value = TRUE)
if (length(file_arg)) setwd(dirname(normalizePath(sub('^--file=', '', file_arg[[1]]))))
repo_root <- normalizePath('../..')
data_dir <- file.path(repo_root, 'source_data/fig3_binding_v20261021')
figure_dir <- normalizePath('../fig')
'''+plot[plot.index('dir.create(figure_dir'):]
    for letter, names in mapping.items():
        for filename in names:
            plot = plot.replace("'"+filename.removesuffix('.csv')+"'", "'"+letter+'_'+panels[letter]+'/'+filename.removesuffix('.csv')+"'")
    (plot_folder/'src/plot_all.R').write_text(plot)
    (plot_folder/'subplot_captions.txt').write_text('Cohort: 718 AF2 cases / 371 systems and 70 apo cases / 38 systems. Unique identity-checked sequence mapping. Four unchanged models; fixed holo reference labels. System-macro estimates and 10000 paired system bootstrap draws.\n\n'+
        '\n\n'.join(letter+'. '+claims[letter]+' '+numeric[letter] for letter in panels)+
        '\n\nDistribution layers: A/C/H show per-system distributions, box IQR and mean bootstrap CI. D cell brackets are system IQR, not CI. E includes all 10%-90% recall targets with tie-inclusive costs. F bubble area is mean inspected residue fraction, fill/number is mean recall; uncertainty is in the additional frontier and source tables. G box distributions and mean CI use common confidence groups. I shows all system/comparator pairs and strict joint wins.\n'+
        '\n\nCore6 panels a-f correspond to full panels A/C/D/E/F/H.\n')
    (OUT/'code').mkdir(exist_ok=True)
    for path in code+[ROOT/'tools/plot_figure3_binding_reliability.R']:
        shutil.copy2(path, OUT/'code'/path.name)
    (OUT/'docs').mkdir(exist_ok=True)
    for path in docs:
        if path.exists():
            shutil.copy2(path, OUT/'docs'/path.name)
    (OUT/'README.md').write_text('''# Figure 3 binding reliability

Main numerical interpretation is in docs/. Figures are in figures/; unchanged predictions and aligned labels in paired_residue_predictions.parquet and CSV.GZ.
All 874 starting cases and every exclusion are listed in pair_mapping_audit.csv. Main tables require unique sequence alignment;
sensitivity_all_* includes deterministic first optimal alignment for ambiguous sequences, without claiming unique correspondence.
Original native state labels are retained alongside the fixed holo reference labels. Every baseline and all budget points remain available.
Code is additive. Source-data repository uses source_data/fig3_binding_v20261021 and GSBind_fig3_binding_v20261021.
Original PDB files and model caches are shared project dependencies, not replicated into this analysis bundle.
''')
    paper = ROOT.parent/'GSBind_RECOMB27/analysis/figure3_binding_reliability_20261021'
    paper.mkdir(exist_ok=True)
    for path in docs+[OUT/'VALIDATION.json', OUT/'CONTRACT.json']:
        shutil.copy2(path, paper/path.name)
    for path in (OUT/'figures').glob('GSBind_figure3*'):
        shutil.copy2(path, paper/path.name)


if __name__=='__main__':
    validate()
    release()
