import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from common import LEGACY, PACKAGE, ROOT, write_json
from gaussian_direct.data import collate_single, compact_cache_path_for_row, load_checkpoint_model, load_compact_npz, model_arguments_from_batch
from gsbind.covariates import structure_features
from gsbind.surface import SURFACE_CONDITIONS, SurfaceContextCutoff


def cached_arguments(folder, device):
    with np.load(folder / 'frozen_gaussian_inputs.npz') as saved:
        tensors = {name: torch.from_numpy(saved[name].copy()).to(device) for name in saved.files if name != 'vectors_present'}
        cache = {name[7:]: value for name, value in tensors.items() if name.startswith('cache__')}
        scalars = json.loads((folder / 'scalar_cache.json').read_text())
        cache.update({name: tuple(value['__tuple__']) if isinstance(value, dict) and '__tuple__' in value else value for name, value in scalars.items()})
        return tensors['residue_state'], tensors['residue_vectors'] if bool(saved['vectors_present']) else None, tensors['residue_centers'], cache


def process(row, model, device, fresh_full=False):
    identifier = row['uniprot']
    destination = ROOT / 'features' / identifier
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'complete.json').exists():
        return json.loads((destination / 'complete.json').read_text())
    surface = PACKAGE / 'datasets/allosteric/surface' / (identifier+'.npz')
    legacy = LEGACY / 'cases' / identifier
    if surface.exists() and not fresh_full:
        with np.load(surface) as saved:
            hidden = saved['fused'].copy()
            logits = saved['logits'].copy()
            conditions = saved['conditions'].tolist()
        origin = 'reused_verified_E11_surface_forward'
    else:
        if (legacy / 'frozen_gaussian_inputs.npz').exists() and not fresh_full:
            arguments = cached_arguments(legacy, device)
            def forward():
                model.gaussian(*arguments)
                return model._direct_protein_output(model.gaussian.last_fused)
            origin = 'E11_cached_atom_inputs_new_surface_forward'
        else:
            compact = Path(row['compact_path']) if row.get('compact_path') else compact_cache_path_for_row(ROOT / 'compact', row)
            if not compact.exists():
                return dict(uniprot=identifier, status='pending_compact')
            item = load_compact_npz(compact, case_id=row['apex_case_id'], labels=row['labels'], label_mask=row['label_mask'], input_state='afdb_monomer')
            arguments = model_arguments_from_batch(collate_single([item]), torch.device(device))
            captured = []
            handle = model.gaussian.register_forward_pre_hook(lambda module,inputs: captured.append(inputs))
            try:
                with torch.inference_mode():
                    native_output = model(**arguments).detach().float().cpu().numpy().copy()
                    native_hidden = model.gaussian.last_fused.detach().float().cpu().numpy().copy()
            finally:
                handle.remove()
            gaussian_arguments = captured[0]
            def forward():
                model.gaussian(*gaussian_arguments)
                return model._direct_protein_output(model.gaussian.last_fused)
            origin = 'full_E11_new_forward'
        logits, hidden = [], []
        conditions = SURFACE_CONDITIONS
        for condition in conditions:
            with torch.inference_mode(), SurfaceContextCutoff(model.gaussian, condition):
                result = forward()
                logits.append(result.detach().float().cpu().numpy())
                hidden.append(model.gaussian.last_fused.detach().float().cpu().numpy())
        logits, hidden = np.asarray(logits), np.asarray(hidden)
        if origin == 'full_E11_new_forward':
            np.testing.assert_array_equal(hidden[0],native_hidden)
            np.testing.assert_array_equal(logits[0],native_output)
    np.testing.assert_array_equal(hidden[0], hidden[1])
    np.testing.assert_array_equal(logits[0], logits[1])
    if hidden.shape != (7, len(row['sequence']), 128):
        raise ValueError('Hidden dimensions/sequence mismatch')
    if (legacy / 'features.parquet').exists():
        frame = pd.read_parquet(legacy / 'features.parquet')
    else:
        try:
            frame = structure_features(row)
        except Exception:
            records = []
            for line in Path(row['receptor_pdb_path']).read_text().splitlines():
                if line.startswith('ENDMDL'):
                    break
                if line.startswith(('ATOM  ', 'HETATM', 'TER   ')):
                    records.append(line)
            repaired = ROOT / 'raw/dssp_inputs' / (identifier + '.pdb')
            repaired.parent.mkdir(exist_ok=True)
            repaired.write_text('HEADER    SINGLE MODEL DSSP INPUT\n' + '\n'.join(records) + '\nEND\n')
            frame = structure_features(dict(row, receptor_pdb_path=str(repaired)))
    if ''.join(frame.amino_acid) != row['sequence']:
        raise ValueError('Structural feature sequence mismatch')
    frame['uniprot'] = identifier
    frame['case_id'] = row['apex_case_id']
    frame['score'] = expit(logits[0, :, 0])
    frame['protein_length'] = len(row['sequence'])
    frame.to_parquet(destination / 'residues.parquet', index=False)
    np.savez_compressed(destination / 'surface.npz', hidden=hidden, logits=logits, conditions=np.asarray(conditions))
    result = dict(uniprot=identifier, status='complete', residues=len(frame), origin=origin, identity_exact=True)
    write_json(destination / 'complete.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--wait-compact', action='store_true')
    parser.add_argument('--manifest', default=str(ROOT / 'manifest.csv'))
    parser.add_argument('--fresh-full', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model, _ = load_checkpoint_model(PACKAGE / 'checkpoints/direct128_e11.pt', args.device)
    model.eval()
    manifest = pd.read_csv(args.manifest, keep_default_na=False).sort_values('uniprot')
    rows = manifest.to_dict('records')[args.shard::args.shards]
    results, pending = [], rows
    for attempt in range(180):
        remaining = []
        for row in pending:
            try:
                result = process(row, model, args.device, args.fresh_full)
            except Exception as error:
                torch.cuda.empty_cache()
                result = dict(uniprot=row['uniprot'], status='failed', reason=type(error).__name__ + ': ' + str(error))
            if result['status'] == 'pending_compact':
                remaining.append(row)
            else:
                results.append(result)
                print(len(results), '/', len(rows), result, flush=True)
        if not remaining or not args.wait_compact:
            results.extend(dict(uniprot=row['uniprot'], status='pending_compact') for row in remaining)
            break
        pending = remaining
        time.sleep(10)
    pd.DataFrame(results).to_csv(ROOT / ('feature_coverage_shard' + str(args.shard) + '.csv'), index=False)


if __name__ == '__main__':
    main()
