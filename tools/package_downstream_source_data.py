"""Package the approved four downstream result panels for the source-data repository."""

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import shutil

import pandas as pd

from gsbind.downstream import METHODS
from gsbind.downstream_figures import COLORS, LABELS, source_tables, violin


ROOT = Path(__file__).resolve().parents[1]
VERSION = 'fig4_downstream_v20261023'
PANELS = [('A', 'AlloBench', 'AlloBench component AUPR distribution'),
          ('B', 'AM', 'Frozen AlloBench-to-AM component AUPR distribution'),
          ('C', 'Src', 'Src functional-sensitivity prediction'),
          ('D', 'pooled_AUPR', 'Pooled trapezoidal AUPR comparison')]


def package(destination):
    code = destination / ('GSBind_' + VERSION)
    data = destination / 'source_data' / VERSION
    for path in [code / 'fig', code / 'src', data / 'shared', data / 'README', data / 'provenance']:
        path.mkdir(parents=True, exist_ok=True)
    tables = source_tables(ROOT)
    for name, table in tables.items():
        table.to_csv(data / 'shared' / (name+'.csv'), index=False)
    for name in ['METRICS.csv', 'DETAILED_METRICS.csv', 'FOLD_METRICS.csv']:
        shutil.copy2(ROOT / 'results/downstream' / name, data / 'shared' / name)
    for task in ['AlloBench', 'AM', 'Src']:
        shutil.copy2(ROOT / 'results/downstream' / task / 'predictions.csv.gz',
                     data / 'shared' / (task+'_predictions.csv.gz'))
    shutil.copy2(ROOT / 'datasets/downstream/AM/TRANSFER_ROUTING.csv', data / 'shared/AM_TRANSFER_ROUTING.csv')
    for source_name, target_name in [
            ('datasets/functional/raw/energy_single_variants.csv', 'Src_single_variant_effects.csv'),
            ('datasets/allosteric/raw/AlloBench.csv', 'AlloBench_source_records.csv'),
            ('datasets/allosteric/allobench_record_mapping.csv', 'AlloBench_record_mapping.csv')]:
        shutil.copy2(ROOT / source_name, data / 'shared' / target_name)
    for task in ['AlloBench', 'Src']:
        for method in METHODS:
            for fold in range(5):
                source = ROOT / 'checkpoints/downstream' / task / method / f'fold{fold}'
                target = data / 'provenance' / task / method / f'fold{fold}'
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source / 'SELECTION.json', target / 'SELECTION.json')
    shutil.copy2(ROOT / 'configs/downstream_selected.json', data / 'provenance/PROTOCOL.json')
    shutil.copy2(ROOT / 'docs/DOWNSTREAM_SELECTED_PROTOCOL_CN.md', data / 'provenance/METHODS_CN.md')
    shutil.copy2(ROOT / 'docs/downstream_references.bib', data / 'provenance/references.bib')
    alignment = []
    for letter, panel, title in PANELS:
        folder = data / (letter+'_'+panel)
        folder.mkdir(exist_ok=True)
        source_name = f'downstream_{letter}_{panel}'
        for suffix in ['png', 'pdf']:
            source = ROOT / 'figures' / (source_name+'.'+suffix)
            shutil.copy2(source, code / 'fig' / (f'fig4{letter.lower()}_{panel}.'+suffix))
            shutil.copy2(source, folder / ('reference.'+suffix))
        if letter in ['A', 'B']:
            rows = tables['component_AUPR']
            rows = rows.loc[rows.task.eq(panel) & rows.scope.eq('all_positions')]
            rows.to_csv(folder / 'component_AUPR.csv', index=False)
            tables['panel_summary'].query('task == @panel and scope == "all_positions"').to_csv(
                folder / 'summary.csv', index=False)
            description = ('Each observation is one homology component. Trapezoidal AUPR is calculated '
                           'per protein and averaged within component. X: four readouts; Y: component AUPR. '
                           'Colored violins show kernel densities; white boxes show quartiles and medians; '
                           'whiskers extend to observed values within 1.5 IQR. No scatter, mean diamonds, '
                           'significance tests or confidence intervals are overlaid. All component values '
                           'are retained. AlloBench has 282 components; AM has 112. '
                           'AlloBench is source-test-selected OOF; AM uses frozen source probes without target fitting. '
                           'Full AM is not completely source-independent.')
            conclusion = 'Hidden-feature readouts have higher component-macro AUPR than the three comparators.'
        elif letter == 'C':
            rows = tables['src_construct_rho']
            rows.to_csv(folder / 'construct_rho.csv', index=False)
            rows.groupby('method', as_index=False).rho.mean().to_csv(folder / 'bar_values.csv', index=False)
            description = ('X: four readouts; Y: arithmetic mean of full-length and kinase-domain complete-OOF '
                           'Spearman correlations, 269 positions per construct. Targets are position-wise mean '
                           'absolute activity effects across measured non-WT substitutions. This is functional '
                           'sensitivity, not activation direction or binary allosteric-pocket detection. '
                           'Predictions are concatenated across folds before calculating each construct correlation; '
                           'bars are not mean fold correlations. No replicate-based error bars. '
                           'Checkpoints are selected on source held-out/test folds: exploratory, not independent-test evidence.')
            conclusion = 'Mean rho is 0.512691 for hidden versus 0.400463 for structure-only; this does not establish causal allostery.'
        else:
            rows = pd.read_csv(ROOT / 'figures/source_data/downstream_selected/pooled_AUPR_bars.csv')
            rows.to_csv(folder / 'bar_values.csv', index=False)
            tables['random_baselines'].to_csv(folder / 'random_baselines.csv', index=False)
            description = ('X: AlloBench and AM, grouped by four readouts; Y: pooled trapezoidal AUPR '
                           '(sklearn.metrics.auc(recall, precision)), not AP and not case/component-macro AUPR. '
                           'Horizontal numbers are actual bar values. Gray dashed segments show dataset-specific '
                           'positive prevalence as the no-information PR reference level, not an exact finite-sample '
                           'expected trapezoidal area under shuffled rankings. Baselines are 0.03421285949 and '
                           '0.46585230428. No checkpoint reselection or invented replicate error bars.')
            conclusion = 'Hidden has the highest pooled AUPR on both datasets; prevalence differs, so dataset heights are not transfer gains.'
        (folder / 'DESCRIPTION.md').write_text('# '+title+'\n\n'+description+'\n')
        (folder / 'CONCLUSION.md').write_text('# Interpretation\n\n'+conclusion+'\n')
        alignment.append(dict(panel=letter, directory=folder.name, claim_and_estimand=title,
                              description=folder.name+'/DESCRIPTION.md', reference=folder.name+'/reference.png'))
    pd.DataFrame(alignment).to_csv(data / 'CLAIM_DATA_ALIGNMENT.csv', index=False)
    for source_name, target_name in [('downstream_selected_transfer', 'GSBind_figure4_ABC'),
                                     ('downstream_results_four_panel', 'GSBind_figure4_ABCD')]:
        for suffix in ['png', 'pdf']:
            shutil.copy2(ROOT / 'figures' / (source_name+'.'+suffix), code / 'fig' / (target_name+'.'+suffix))
            shutil.copy2(ROOT / 'figures' / (source_name+'.'+suffix), data / (target_name+'.'+suffix))
    support = ('import numpy as np\n\nMETHODS = '+repr(METHODS)+'\nLABELS = '+repr(LABELS)+
               '\nCOLORS = '+repr(COLORS)+'\n\n'+inspect.getsource(violin))
    (code / 'src/plot_support.py').write_text(support)
    rendering = (ROOT / 'src/gsbind/downstream_publication.py').read_text()
    rendering = rendering.replace('from .downstream import METHODS', 'from plot_support import METHODS')
    rendering = rendering.replace('from .downstream_figures import COLORS, LABELS, violin',
                                  'from plot_support import COLORS, LABELS, violin')
    (code / 'src/publication_panels.py').write_text(rendering)
    for name in ['plot_all.py', 'validate.py']:
        shutil.copy2(ROOT / 'tools/downstream_source_templates' / name, code / 'src' / name)
    (code / 'README.md').write_text(
        '# Figure 4 downstream results — 2026-10-23\n\n'
        'ABC are the approved AlloBench/AM violins and Src bars. D is pooled AUPR with prevalence baselines '
        'and horizontal numeric labels. Two structural cases are intentionally not part of this upload.\n\n'
        'Figures: `fig/GSBind_figure4_ABC.png`, `fig/GSBind_figure4_ABCD.png`, and `fig/fig4a_*` through '
        '`fig/fig4d_*` (PNG/PDF). Data: `../source_data/'+VERSION+'`.\n\n'
        'From repository root: `python3 GSBind_'+VERSION+'/src/plot_all.py`; '
        'audit: `python3 GSBind_'+VERSION+'/src/validate.py`. '
        'Dependencies: numpy, pandas, scipy, scikit-learn, matplotlib. No training, GPU or external workspace required.\n')
    (data / 'README.md').write_text(
        '# Figure 4 downstream source data\n\n'
        'Panel navigation: `README/panels.md`. Each panel has DESCRIPTION.md, CONCLUSION.md, exact CSVs '
        'and reference PNG/PDF. `shared/` includes all saved residue labels, predictions, case IDs, components '
        'and folds, complete reported metrics, per-protein AUPR, per-construct correlations, AM routing, '
        'and baseline counts. CSV.GZ means ordinary gzip-compressed CSV, not Git LFS.\n\n'
        'A/B: component-macro distributions, not pooled or case-macro. C: two construct correlations averaged, '
        'not fold means. D: pooled trapezoidal AUPR, not AP. The discussed case-macro alternative was not '
        'approved as a replacement and is not silently substituted. `provenance/` records the protocol and '
        'all 40 source-fold checkpoint-selection records; this is a figure/data release, not a checkpoint '
        'or full retraining-feature release. Historic absolute paths in metadata are provenance only and '
        'are not required by plotting or validation.\n\n'
        'AlloBench/Src are exploratory source-test-selected OOF; AM has no target fitting/selection but '
        'full AM is not fully source-independent. Functional sensitivity does not establish activation '
        'direction or causal allostery. Old Fig4 versions are retained, not additional current panels.\n\n'
        'Plot and validate using `../../GSBind_'+VERSION+'/src/plot_all.py` and `validate.py`. '
        'FILE_INDEX.csv lists package files with byte sizes and SHA-256; CODE_HASHES.csv indexes code.\n')
    (data / 'README/panels.md').write_text('# Panel navigation\n\n'+'\n'.join(
        f'- [{letter}: {title}](../{letter}_{panel}/DESCRIPTION.md) — [CSV and reference](../{letter}_{panel}/)'
        for letter, panel, title in PANELS)+'\n')
    (data / 'DATA_DICTIONARY.md').write_text(
        '# Data dictionary and logical keys\n\n'
        '- `shared/AlloBench_predictions.csv.gz`, `AM_predictions.csv.gz`: one row per '
        '`case_id,residue_index_0`; label is binary, residue_index_0 is zero-based, uniprot identifies '
        'a protein and component a homology group. Four `*_probe`/`structure_only` columns are '
        'continuous saved readout scores; raw-score columns are provenance comparators, not extra plotted bars.\n'
        '- AM outer_fold is retained provenance, not an AM training fold. eligible_source_heads and '
        'source_component_disjoint describe frozen transfer routing; see AM_TRANSFER_ROUTING.csv.\n'
        '- `shared/Src_predictions.csv.gz`: one row per case_id,physical_position; physical_position '
        'is one-based human Src numbering. functional_magnitude is the mean absolute measured activity '
        'effect, not its direction. FL and KD share held-out physical positions. See raw '
        'Src_single_variant_effects.csv (context,position,mutant; activity/folding effects and uncertainties).\n'
        '- A/B component_AUPR.csv: task,scope,method,component is the logical key; AUPR is the '
        'mean of per-protein trapezoidal PR integrals within the component. Summary mean_AUPR '
        'averages components equally; median_AUPR is separate.\n'
        '- C construct_rho.csv: case_id,interaction,method is the key; rho is complete-OOF '
        'Spearman, positions is the number evaluated. bar_values.csv averages the two constructs equally.\n'
        '- D bar_values.csv: task,method is the key; pooled_AUPR concatenates evaluation residues. '
        'random_baselines.csv: task is the key; prevalence = positives/positions.\n'
        '- DETAILED_METRICS.csv contains classification protein AUPRs and functional construct rhos '
        'in one table; cross-task inapplicable columns are intentionally empty, not zero. '
        'FOLD_METRICS.csv describes source-fold selection, not the estimator used for C bars.\n'
        '- AlloBench_source_records.csv and AlloBench_record_mapping.csv preserve original annotation '
        'records and mappings; prediction labels and saved evaluation membership are authoritative for '
        'reproducing these panels. No unannotated AM positions are added as negatives.\n'
        '- provenance/ contains the frozen source protocol and per-fold selection metadata. '
        'CSV_COMPLETENESS_AUDIT.csv records dimensions, missing cells and exact duplicate rows; '
        'missing raw annotations are not silently imputed.\n')
    audit = []
    for path in sorted(data.rglob('*')):
        if path.is_file() and (path.suffix == '.csv' or path.name.endswith('.csv.gz')):
            frame = pd.read_csv(path)
            audit.append(dict(relative_path=str(path.relative_to(data)), rows=len(frame),
                              columns=len(frame.columns), missing_cells=int(frame.isna().sum().sum()),
                              duplicate_physical_rows=int(frame.duplicated().sum()),
                              missing_explanation='Not-applicable cross-task fields and unavailable original source annotations are retained, not imputed.'
                              if frame.isna().any().any() else 'None'))
    pd.DataFrame(audit).to_csv(data / 'CSV_COMPLETENESS_AUDIT.csv', index=False)
    (code / 'subplot_captions.txt').write_text('\n\n'.join(
        letter+'. '+(data / (letter+'_'+panel) / 'DESCRIPTION.md').read_text().split('\n\n', 1)[1].strip()
        for letter, panel, _ in PANELS)+'\n')
    code_hashes = [dict(path=str(path.relative_to(destination)), bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                   for path in sorted(code.rglob('*')) if path.is_file() and '__pycache__' not in path.parts]
    pd.DataFrame(code_hashes).to_csv(data / 'CODE_HASHES.csv', index=False)
    index = [dict(relative_path=str(path.relative_to(data)), bytes=path.stat().st_size,
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
             for path in sorted(data.rglob('*')) if path.is_file() and path.name not in ['FILE_INDEX.csv', 'VALIDATION.json']]
    pd.DataFrame(index).to_csv(data / 'FILE_INDEX.csv', index=False)
    print(json.dumps(dict(code=str(code), data=str(data), indexed_files=len(index)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    package(parser.parse_args().destination)
