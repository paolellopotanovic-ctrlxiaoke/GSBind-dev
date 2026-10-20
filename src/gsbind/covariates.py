import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


MAX_ASA = dict(zip('ARNDCQEGHILKMFPSTWYV', [129, 274, 195, 193, 167, 225, 223, 104, 224, 197, 201, 236, 224, 240, 159, 155, 172, 285, 263, 174]))


def structure_features(row):
    executable = shutil.which('dssp') or shutil.which('mkdssp')
    if executable is None:
        raise FileNotFoundError('Install DSSP 4.x (dssp or mkdssp) to rebuild structural controls')
    source = Path(row['receptor_pdb_path'])
    records = [line for line in source.read_text().splitlines() if line[:6].strip() not in ['DBREF', 'DBREF1', 'DBREF2']]
    text = '\n'.join(records) + '\n'
    text = text if text.startswith('HEADER') else 'HEADER    TEMP\n' + text
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory) / 'input.pdb'
        temporary.write_text(text)
        result = subprocess.run([executable, '--output-format', 'dssp', str(temporary)], capture_output=True, text=True, check=True, timeout=120)
    lines = result.stdout.splitlines()
    header = next(index for index, line in enumerate(lines) if line.strip().startswith('#  RESIDUE AA'))
    dssp = {}
    for line in lines[header+1:]:
        if len(line) < 38 or not line[:5].strip().isdigit() or line[11].strip() != row['receptor_chain_id'] or line[10].strip():
            continue
        amino = 'C' if line[13].islower() else line[13]
        if amino not in MAX_ASA:
            continue
        code = line[16]
        dssp[int(line[5:10])] = dict(amino_acid=amino, relative_sasa=float(line[35:38])/MAX_ASA[amino],
                                    dssp_helix=int(code in 'HIG'), dssp_strand=int(code in 'EB'), dssp_turn=int(code in 'TS'))
    coordinates, confidence = {}, {}
    for line in source.read_text().splitlines():
        if line.startswith('ENDMDL'):
            break
        if line.startswith('ATOM  ') and line[12:16].strip() == 'CA' and line[21] == row['receptor_chain_id'] and line[16] in ' A' and not line[26].strip():
            number = int(line[22:26])
            coordinates.setdefault(number, [float(line[30:38]), float(line[38:46]), float(line[46:54])])
            confidence.setdefault(number, float(line[60:66]))
    numbers = [int(value) for value in row['input_residue_ids'].split(';')]
    geometry = np.asarray([coordinates[number] for number in numbers])
    packing = (cdist(geometry, geometry) <= 8).sum(axis=1) - 1
    output = []
    for index, (number, amino) in enumerate(zip(numbers, row['sequence'])):
        if number not in dssp or dssp[number]['amino_acid'] != amino:
            raise ValueError(f'DSSP identity mismatch {row["apex_case_id"]} {number}')
        output.append(dict(case_id=row['apex_case_id'], residue_index_0=index, protein_length=len(numbers),
                           **dssp[number], residue_plddt=confidence[number], contact_density_8A=int(packing[index])))
    return pd.DataFrame(output)


def rebuild(root, manifest, output):
    frames = [structure_features(row) for row in pd.read_csv(manifest, keep_default_na=False).to_dict('records')]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(output, index=False)
