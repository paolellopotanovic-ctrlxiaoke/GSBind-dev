import numpy as np
import pandas as pd


def centered_hidden(frame, hidden):
    frame = frame.reset_index(drop=True)
    hidden = np.asarray(hidden)
    if hidden.shape != (len(frame), 128) or not np.isfinite(hidden).all():
        raise ValueError('Require aligned complete E11 [residues,128] hidden')
    result = hidden.copy()
    for case_id, group in frame.groupby('case_id'):
        if not np.array_equal(np.sort(group.residue_index_0.to_numpy()), np.arange(int(group.protein_length.iloc[0]))):
            raise ValueError('Complete protein required: ' + str(case_id))
        low = group.index[group.score.lt(.5)]
        selected = low if len(low) else group.index
        result[group.index] = hidden[group.index] - pd.DataFrame(hidden[selected]).mean().to_numpy(dtype=np.float32)
    return result


def training_weights(frame):
    positions = frame.groupby('uniprot').size()
    proteins = frame[['component', 'uniprot']].drop_duplicates().groupby('component').size()
    weights = np.asarray([1 / (positions[row.uniprot] * proteins[row.component]) for row in frame.itertuples()])
    return weights / weights.mean()
