import csv
import random
import sys
from pathlib import Path

import torch as pt

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / "model"
for path in (str(ROOT_DIR), str(MODEL_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.data_encoding import encode_features, encode_structure, extract_topology
from src.dataset import collate_batch_features
from src.structure import clean_structure
from src.structure_io import read_pdb

from pinder_pesto import STATE_TO_ID


FIELD_SEP = ";"
AA_TO_IDX = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
    "X": 20,
}
PAD_IDX = 21


def read_manifest(path):
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_field(text, cast=str):
    if text is None or text == "":
        return []
    return [cast(x) for x in str(text).split(FIELD_SEP)]


def sequence_to_tensor(sequence):
    indices = [AA_TO_IDX.get(aa.upper(), AA_TO_IDX["X"]) for aa in str(sequence or "").strip()]
    if not indices:
        indices = [PAD_IDX]
    return pt.tensor(indices, dtype=pt.long)


def sequence_mask(sequence_tensor):
    return (sequence_tensor != PAD_IDX).to(pt.float)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def filter_manifest_rows(rows, states=None, require_usable=True):
    states = set(states or STATE_TO_ID)
    out = []
    for row in rows:
        if row.get("input_state") not in states:
            continue
        if require_usable:
            if str(row.get("low_mapping_quality", "0")) == "1":
                continue
            if int(float(row.get("num_positive_labels", 0))) <= 0:
                continue
        out.append(row)
    return out


def split_by_system(rows, val_fraction=0.2, seed=42):
    systems = sorted({row["system_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(systems)
    n_val = max(1, int(round(len(systems) * val_fraction))) if len(systems) > 1 else 0
    val_systems = set(systems[:n_val])
    train_rows = [row for row in rows if row["system_id"] not in val_systems]
    val_rows = [row for row in rows if row["system_id"] in val_systems]
    return train_rows, val_rows


class PinderManifestDataset(pt.utils.data.Dataset):
    def __init__(self, rows, max_num_nn=64):
        self.rows = list(rows)
        self.max_num_nn = int(max_num_nn)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        pdb_path = row.get("receptor_pdb_path") or row.get("input_pdb_path")
        structure = clean_structure(read_pdb(pdb_path))
        X, M = encode_structure(structure)
        q = encode_features(structure)[0]
        ids_topk = extract_topology(X, self.max_num_nn)[0]

        labels = pt.tensor(split_field(row["labels"], int), dtype=pt.float)
        mask = pt.tensor(split_field(row["label_mask"], int), dtype=pt.float)
        n_res = M.shape[1]
        if labels.numel() < n_res:
            labels = pt.cat([labels, pt.zeros(n_res - labels.numel())])
            mask = pt.cat([mask, pt.zeros(n_res - mask.numel())])
        elif labels.numel() > n_res:
            labels = labels[:n_res]
            mask = mask[:n_res]

        state_id = STATE_TO_ID.get(row.get("input_state", "apo"), STATE_TO_ID["apo"])
        state_ids = pt.full((n_res,), state_id, dtype=pt.long)
        partner_seq_indices = sequence_to_tensor(row.get("partner_sequence", ""))
        partner_mask = sequence_mask(partner_seq_indices)
        return {
            "X": X,
            "ids_topk": ids_topk,
            "q": q,
            "M": M.float(),
            "y": labels,
            "mask": mask,
            "state_ids": state_ids,
            "partner_seq_indices": partner_seq_indices,
            "partner_mask": partner_mask,
            "row": row,
        }


def collate_pinder_batch(batch):
    X, ids_topk, q, M = collate_batch_features(
        [[item["X"], item["ids_topk"], item["q"], item["M"]] for item in batch]
    )
    y = pt.cat([item["y"] for item in batch], dim=0)
    mask = pt.cat([item["mask"] for item in batch], dim=0)
    state_ids = pt.cat([item["state_ids"] for item in batch], dim=0)
    partner_seq_indices = pt.nn.utils.rnn.pad_sequence(
        [item["partner_seq_indices"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    partner_mask = pt.nn.utils.rnn.pad_sequence(
        [item["partner_mask"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    rows = [item["row"] for item in batch]
    return {
        "X": X,
        "ids_topk": ids_topk,
        "q": q,
        "M": M,
        "y": y,
        "mask": mask,
        "state_ids": state_ids,
        "partner_seq_indices": partner_seq_indices,
        "partner_mask": partner_mask,
        "rows": rows,
    }
