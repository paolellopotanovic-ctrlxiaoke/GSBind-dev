import csv
import hashlib
import sys
from functools import lru_cache
from pathlib import Path

import torch as pt

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_encoding import encode_features, encode_structure, extract_topology
from src.dataset import collate_batch_features
from src.structure import clean_structure, res3to1
from src.structure_io import read_pdb


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


def residue_uid(resid, icode=""):
    icode = str(icode or "").strip()
    return f"{int(resid)}{icode}" if icode else str(int(resid))


def structure_residue_ids(structure):
    ids = []
    seen = set()
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    for chain, resid, icode in zip(structure["chain_name"].tolist(), structure["resid"].tolist(), icodes):
        chain = str(chain).split(":")[0]
        key = (chain, int(resid), str(icode or "").strip())
        if key in seen:
            continue
        seen.add(key)
        ids.append(residue_uid(resid, icode))
    return ids


def structure_residue_chain_ids(structure):
    chain_to_idx = {}
    ids = []
    seen = set()
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    for chain, resid, icode in zip(structure["chain_name"].tolist(), structure["resid"].tolist(), icodes):
        chain = str(chain).split(":")[0]
        key = (chain, int(resid), str(icode or "").strip())
        if key in seen:
            continue
        seen.add(key)
        if chain not in chain_to_idx:
            chain_to_idx[chain] = len(chain_to_idx)
        ids.append(chain_to_idx[chain])
    return pt.tensor(ids, dtype=pt.long)


@lru_cache(maxsize=8192)
def residue_chain_ids_for_pdb(path):
    try:
        structure = clean_structure(read_pdb(str(path)))
        return structure_residue_chain_ids(structure).cpu()
    except Exception:
        return None


def raw_residue_ids_and_sequence(path):
    structure = read_pdb(str(path))
    ids = []
    seq = []
    seen = set()
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    for i, resid in enumerate(structure["resid"]):
        if str(structure["het_flag"][i]) != "A":
            continue
        chain = str(structure["chain_name"][i]).split(":")[0]
        icode = str(icodes[i] or "").strip()
        key = (chain, int(resid), icode)
        if key in seen:
            continue
        seen.add(key)
        ids.append(residue_uid(resid, icode))
        seq.append(res3to1.get(str(structure["resname"][i]).strip().upper(), "X"))
    return ids, seq


def sequence_to_tensor(seq):
    ids = [AA_TO_IDX.get(aa.upper(), AA_TO_IDX["X"]) for aa in str(seq).strip()]
    if not ids:
        ids = [PAD_IDX]
    return pt.tensor(ids, dtype=pt.long)


def sequence_mask(seq_tensor):
    return (seq_tensor != PAD_IDX).to(pt.float)


AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def clean_esm_sequence(seq, max_length=1022):
    seq = "".join((seq or "").strip().upper().split())
    seq = "".join(ch if ch in AA20 else "X" for ch in seq)
    if not seq:
        seq = "X"
    if max_length and len(seq) > max_length:
        seq = seq[:max_length]
    return seq


def esm_sequence_key(sequence, model_name="esm2_t33_650M_UR50D", repr_layer=33, max_length=1022):
    seq = clean_esm_sequence(sequence, max_length=max_length)
    payload = f"{model_name}|layer={repr_layer}|max={max_length}|seq={seq}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def esm_cache_path_for_sequence(
    cache_dir,
    sequence,
    model_name="esm2_t33_650M_UR50D",
    repr_layer=33,
    max_length=1022,
):
    key = esm_sequence_key(sequence, model_name=model_name, repr_layer=repr_layer, max_length=max_length)
    return Path(cache_dir) / key[:2] / f"{key}.pt"


def esm_cache_path_for_key(cache_dir, key):
    key = str(key).strip()
    return Path(cache_dir) / key[:2] / f"{key}.pt"


def structure_sequence_to_tensor(structure):
    seq = []
    seen = set()
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    for chain, resid, icode, resname in zip(
        structure["chain_name"].tolist(),
        structure["resid"].tolist(),
        icodes,
        structure["resname"].tolist(),
    ):
        key = (str(chain).split(":")[0], int(resid), str(icode or "").strip())
        if key in seen:
            continue
        seen.add(key)
        seq.append(res3to1.get(str(resname).upper(), "X"))
    return sequence_to_tensor("".join(seq))


def structure_sequence_list(structure):
    seq = []
    seen = set()
    icodes = structure.get("icode", [""] * len(structure["resid"]))
    for chain, resid, icode, resname in zip(
        structure["chain_name"].tolist(),
        structure["resid"].tolist(),
        icodes,
        structure["resname"].tolist(),
    ):
        key = (str(chain).split(":")[0], int(resid), str(icode or "").strip())
        if key in seen:
            continue
        seen.add(key)
        seq.append(res3to1.get(str(resname).strip().upper(), "X"))
    return seq


def cache_key_for_row(row):
    raw = FIELD_SEP.join(
        [
            row.get("apex_case_id", ""),
            row.get("receptor_pdb_path", ""),
            row.get("partner_sequence", ""),
            row.get("input_residue_ids", ""),
            row.get("labels", ""),
            row.get("label_mask", ""),
        ]
    )
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in row.get("apex_case_id", "apex"))[:120]
    return f"{stem}_{digest}.pt"


def cache_path_for_row(cache_dir, row):
    return Path(cache_dir) / cache_key_for_row(row)


def graph_cache_key_for_row(row, ks):
    raw = "|".join([cache_key_for_row(row), ",".join(str(int(k)) for k in ks)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20] + ".pt"


def graph_cache_path_for_row(cache_dir, row, ks):
    return Path(cache_dir) / graph_cache_key_for_row(row, ks)


def build_apex_tensor_item(row, max_num_nn=64, strict_residue_alignment=True, skip_bad_examples=False):
    structure = clean_structure(read_pdb(row["receptor_pdb_path"]))
    X, M = encode_structure(structure)
    q = pt.cat(encode_features(structure), dim=1)
    ids_topk = extract_topology(X, int(max_num_nn))[0]
    receptor_seq = structure_sequence_to_tensor(structure)
    residue_chain_ids = structure_residue_chain_ids(structure)

    labels = pt.tensor(split_field(row.get("labels", ""), int), dtype=pt.float)
    mask = pt.tensor(split_field(row.get("label_mask", ""), int), dtype=pt.float)
    n_res = M.shape[1]
    if receptor_seq.numel() < n_res:
        receptor_seq = pt.cat([receptor_seq, pt.full((n_res - receptor_seq.numel(),), AA_TO_IDX["X"], dtype=pt.long)])
    elif receptor_seq.numel() > n_res:
        receptor_seq = receptor_seq[:n_res]
    if strict_residue_alignment:
        if labels.numel() != n_res or mask.numel() != n_res:
            message = (
                f"Residue count mismatch for apex_case_id={row.get('apex_case_id', '')}: "
                f"labels={labels.numel()} mask={mask.numel()} encoded={n_res}"
            )
            if skip_bad_examples:
                return {
                    "skip": True,
                    "skip_reason": message,
                    "rows": row,
                }
            raise ValueError(message)
        manifest_residue_ids = split_field(row.get("input_residue_ids", ""), str)
        if manifest_residue_ids:
            raw_ids, raw_seq = raw_residue_ids_and_sequence(row["receptor_pdb_path"])
            clean_seq = structure_sequence_list(structure)
        if manifest_residue_ids and (
            manifest_residue_ids != raw_ids or len(raw_ids) != n_res or raw_seq != clean_seq
        ):
            message = (
                f"Ordinal residue alignment mismatch for apex_case_id={row.get('apex_case_id', '')}: "
                f"manifest={len(manifest_residue_ids)} raw={len(raw_ids)} encoded={n_res}"
            )
            if skip_bad_examples:
                return {
                    "skip": True,
                    "skip_reason": message,
                    "rows": row,
                }
            raise ValueError(message)
    if labels.numel() < n_res:
        labels = pt.cat([labels, pt.zeros(n_res - labels.numel())])
        mask = pt.cat([mask, pt.zeros(n_res - mask.numel())])
    elif labels.numel() > n_res:
        labels = labels[:n_res]
        mask = mask[:n_res]

    partner_seq = sequence_to_tensor(row["partner_sequence"])
    partner_mask = sequence_mask(partner_seq)

    return {
        "X": X,
        "ids_topk": ids_topk,
        "q": q,
        "M": M.float(),
        "y": labels,
        "mask": mask,
        "state_ids": pt.full((n_res,), 0 if row.get("input_state") == "holo" else 1, dtype=pt.long),
        "receptor_seq_indices": receptor_seq,
        "receptor_mask": sequence_mask(receptor_seq),
        "partner_seq_indices": partner_seq,
        "partner_mask": partner_mask,
        "residue_chain_ids": residue_chain_ids[:n_res] if residue_chain_ids.numel() >= n_res else pt.zeros(n_res, dtype=pt.long),
    }


def save_apex_tensor_item(path, item):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_item = {k: v.cpu() if isinstance(v, pt.Tensor) else v for k, v in item.items() if k != "rows"}
    pt.save(tensor_item, path)


class ApexPairedManifestDataset(pt.utils.data.Dataset):
    """Paired apo/AF2 receptor + partner-sequence dataset for APEX-Net."""

    def __init__(
        self,
        rows,
        max_num_nn=64,
        strict_residue_alignment=True,
        skip_bad_examples=False,
        tensor_cache_dir=None,
        require_tensor_cache=False,
        esm_cache_dir=None,
        require_esm_cache=False,
        esm_model_name="esm2_t33_650M_UR50D",
        esm_repr_layer=33,
        esm_max_length=1022,
        residue_graph_cache_dir=None,
        require_residue_graph_cache=False,
        residue_graph_ks=(16, 24, 32),
        receptor_esm_cache_dir=None,
    ):
        self.rows = list(rows)
        self.max_num_nn = int(max_num_nn)
        self.strict_residue_alignment = bool(strict_residue_alignment)
        self.skip_bad_examples = bool(skip_bad_examples)
        self.tensor_cache_dir = Path(tensor_cache_dir) if tensor_cache_dir else None
        self.require_tensor_cache = bool(require_tensor_cache)
        self.esm_cache_dir = Path(esm_cache_dir) if esm_cache_dir else None
        self.require_esm_cache = bool(require_esm_cache)
        self.esm_model_name = str(esm_model_name)
        self.esm_repr_layer = int(esm_repr_layer)
        self.esm_max_length = int(esm_max_length)
        self.receptor_esm_cache_dir = Path(receptor_esm_cache_dir) if receptor_esm_cache_dir else None
        self.residue_graph_cache_dir = Path(residue_graph_cache_dir) if residue_graph_cache_dir else None
        self.require_residue_graph_cache = bool(require_residue_graph_cache)
        self.residue_graph_ks = tuple(int(k) for k in residue_graph_ks)

    def _attach_partner_esm_cache(self, item, row):
        if self.esm_cache_dir is None:
            return item
        path = esm_cache_path_for_sequence(
            self.esm_cache_dir,
            row.get("partner_sequence", ""),
            model_name=self.esm_model_name,
            repr_layer=self.esm_repr_layer,
            max_length=self.esm_max_length,
        )
        if not path.is_file():
            message = f"Missing ESM cache for apex_case_id={row.get('apex_case_id', '')}: {path}"
            if self.require_esm_cache or self.skip_bad_examples:
                return {"skip": True, "skip_reason": message, "rows": row}
            return item
        cache = pt.load(path, map_location="cpu")
        emb = cache["embedding"].to(dtype=pt.float)
        item["partner_esm_embedding"] = emb
        item["partner_esm_mask"] = pt.ones((emb.shape[0],), dtype=pt.float)
        return item

    def _attach_receptor_esm_cache(self, item, row):
        if self.receptor_esm_cache_dir is None:
            return item
        cid = row.get("apex_case_id", "").strip()
        if not cid:
            return item
        # Load hash mapping (cached in memory)
        if not hasattr(self, "_receptor_esm_hash_map"):
            map_file = self.receptor_esm_cache_dir / "case_id_map.json"
            if map_file.is_file():
                import json
                with open(map_file) as f:
                    self._receptor_esm_hash_map = json.load(f)
            else:
                self._receptor_esm_hash_map = {}
        hash_id = self._receptor_esm_hash_map.get(cid)
        if not hash_id:
            return item
        path = self.receptor_esm_cache_dir / f"{hash_id}.pt"
        if not path.is_file():
            return item
        try:
            emb = pt.load(path, map_location="cpu")
            if emb.dtype != pt.float32:
                emb = emb.to(dtype=pt.float32)
            item["receptor_esm_embedding"] = emb
            item["receptor_esm_mask"] = pt.ones((emb.shape[0],), dtype=pt.float)
        except Exception:
            pass
        return item

    def _attach_residue_graph_cache(self, item, row):
        if self.residue_graph_cache_dir is None:
            return item
        path = graph_cache_path_for_row(self.residue_graph_cache_dir, row, self.residue_graph_ks)
        if not path.is_file():
            message = f"Missing residue graph cache for apex_case_id={row.get('apex_case_id', '')}: {path}"
            if self.require_residue_graph_cache or self.skip_bad_examples:
                return {"skip": True, "skip_reason": message, "rows": row}
            return item
        cache = pt.load(path, map_location="cpu")
        n_res = int(item["M"].shape[1])
        if int(cache.get("num_residues", -1)) != n_res:
            message = (
                f"Residue graph cache size mismatch for apex_case_id={row.get('apex_case_id', '')}: "
                f"cache={cache.get('num_residues')} tensor={n_res}"
            )
            if self.require_residue_graph_cache or self.skip_bad_examples:
                return {"skip": True, "skip_reason": message, "rows": row}
            return item
        item["residue_graph_cache"] = cache
        return item

    def _attach_residue_chain_ids(self, item, row):
        if item.get("skip") or "residue_chain_ids" in item:
            return item
        n_res = int(item["M"].shape[1])
        chain_ids = residue_chain_ids_for_pdb(row.get("receptor_pdb_path", ""))
        if isinstance(chain_ids, pt.Tensor) and int(chain_ids.numel()) >= n_res:
            item["residue_chain_ids"] = chain_ids[:n_res].long()
        else:
            item["residue_chain_ids"] = pt.zeros(n_res, dtype=pt.long)
        return item

    def _attach_optional_caches(self, item, row):
        item = self._attach_residue_chain_ids(item, row)
        if item.get("skip"):
            return item
        item = self._attach_partner_esm_cache(item, row)
        if item.get("skip"):
            return item
        item = self._attach_receptor_esm_cache(item, row)
        if item.get("skip"):
            return item
        item = self._attach_residue_graph_cache(item, row)
        return item

    def _attach_manifest_labels(self, item, row):
        if str(row.get("tensor_cache_labels_from_manifest", "")).lower() not in {"1", "true", "yes"}:
            return item
        labels = pt.tensor(split_field(row.get("labels", ""), int), dtype=pt.float)
        mask = pt.tensor(split_field(row.get("label_mask", ""), int), dtype=pt.float)
        num_residues = int(item["M"].shape[1])
        if labels.numel() != num_residues or mask.numel() != num_residues:
            message = (
                f"Manifest label overlay mismatch for apex_case_id={row.get('apex_case_id', '')}: "
                f"labels={labels.numel()} mask={mask.numel()} encoded={num_residues}"
            )
            if self.skip_bad_examples:
                return {"skip": True, "skip_reason": message, "rows": row}
            raise ValueError(message)
        item["y"] = labels
        item["mask"] = mask
        partner_seq = sequence_to_tensor(row.get("partner_sequence", ""))
        item["partner_seq_indices"] = partner_seq
        item["partner_mask"] = sequence_mask(partner_seq)
        item["state_ids"] = pt.full(
            (num_residues,),
            0 if row.get("input_state") == "holo" else 1,
            dtype=pt.long,
        )
        return item

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.tensor_cache_dir is not None:
            cache_path = cache_path_for_row(self.tensor_cache_dir, row)
            if cache_path.is_file():
                try:
                    item = pt.load(cache_path, map_location="cpu")
                except Exception as exc:
                    message = (
                        f"Failed to load tensor cache for apex_case_id={row.get('apex_case_id', '')}: "
                        f"{cache_path} ({type(exc).__name__}: {exc})"
                    )
                    if self.skip_bad_examples:
                        return {
                            "skip": True,
                            "skip_reason": message,
                            "rows": row,
                        }
                    raise
                item = self._attach_manifest_labels(item, row)
                if item.get("skip"):
                    return item
                item["rows"] = row
                item = self._attach_optional_caches(item, row)
                return item
            if self.require_tensor_cache:
                message = (
                    f"Missing tensor cache for apex_case_id={row.get('apex_case_id', '')}: {cache_path}"
                )
                if self.skip_bad_examples:
                    return {
                        "skip": True,
                        "skip_reason": message,
                        "rows": row,
                    }
                raise ValueError(message)
        item = build_apex_tensor_item(
            row,
            max_num_nn=self.max_num_nn,
            strict_residue_alignment=self.strict_residue_alignment,
            skip_bad_examples=self.skip_bad_examples,
        )
        if not item.get("skip"):
            item = self._attach_manifest_labels(item, row)
        if not item.get("skip"):
            item["rows"] = row
            item = self._attach_optional_caches(item, row)
        return item


def collate_apex_batch(batch):
    skipped = [item for item in batch if item.get("skip")]
    batch = [item for item in batch if not item.get("skip")]
    if not batch:
        return {
            "skip_batch": True,
            "skipped_count": len(skipped),
            "skipped_rows": [item.get("rows", {}) for item in skipped],
            "skip_reasons": [item.get("skip_reason", "") for item in skipped],
        }
    X, ids_topk, q, M = collate_batch_features(
        [[item["X"], item["ids_topk"], item["q"], item["M"]] for item in batch]
    )
    atom_counts = pt.tensor([item["X"].shape[0] for item in batch], dtype=pt.long)
    residue_counts = pt.tensor([item["M"].shape[1] for item in batch], dtype=pt.long)
    y = pt.cat([item["y"] for item in batch], dim=0)
    mask = pt.cat([item["mask"] for item in batch], dim=0)
    state_ids = pt.cat([item["state_ids"] for item in batch], dim=0)
    residue_chain_ids = pt.cat([item.get("residue_chain_ids", pt.zeros(item["M"].shape[1], dtype=pt.long)) for item in batch], dim=0)
    rows = [item["rows"] for item in batch]

    receptor_seq = pt.nn.utils.rnn.pad_sequence(
        [item["receptor_seq_indices"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    receptor_mask = pt.nn.utils.rnn.pad_sequence(
        [item["receptor_mask"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    partner_seq = pt.nn.utils.rnn.pad_sequence(
        [item["partner_seq_indices"] for item in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    partner_mask = pt.nn.utils.rnn.pad_sequence(
        [item["partner_mask"] for item in batch],
        batch_first=True,
        padding_value=0.0,
    )
    out = {
        "X": X,
        "ids_topk": ids_topk,
        "q": q,
        "M": M,
        "atom_counts": atom_counts,
        "residue_counts": residue_counts,
        "y": y,
        "mask": mask,
        "state_ids": state_ids,
        "residue_chain_ids": residue_chain_ids,
        "receptor_seq_indices": receptor_seq,
        "receptor_mask": receptor_mask,
        "partner_seq_indices": partner_seq,
        "partner_mask": partner_mask,
        "rows": rows,
        "skip_batch": False,
        "skipped_count": len(skipped),
        "skipped_rows": [item.get("rows", {}) for item in skipped],
        "skip_reasons": [item.get("skip_reason", "") for item in skipped],
    }
    if all("receptor_esm_embedding" in item for item in batch):
        out["receptor_esm_embedding"] = pt.nn.utils.rnn.pad_sequence(
            [item["receptor_esm_embedding"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
    if any("partner_esm_embedding" in item for item in batch):
        if not all("partner_esm_embedding" in item for item in batch):
            raise ValueError("Mixed batch with partial partner_esm_embedding cache is not supported.")
        out["partner_esm_embedding"] = pt.nn.utils.rnn.pad_sequence(
            [item["partner_esm_embedding"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
        out["partner_esm_mask"] = pt.nn.utils.rnn.pad_sequence(
            [item["partner_esm_mask"] for item in batch],
            batch_first=True,
            padding_value=0.0,
        )
    if any("residue_graph_cache" in item for item in batch):
        if not all("residue_graph_cache" in item for item in batch):
            raise ValueError("Mixed batch with partial residue_graph_cache is not supported.")
        graph_caches = [item["residue_graph_cache"] for item in batch]
        out["residue_graph_caches"] = graph_caches
    return out
