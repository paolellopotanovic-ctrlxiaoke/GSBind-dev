import argparse
import csv
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_encoding import encode_structure
from src.structure import clean_structure, res3to1
from src.structure_io import read_pdb


FIELD_SEP = ";"


def split_field(text):
    if text is None or text == "":
        return []
    return [x for x in str(text).split(FIELD_SEP) if x != ""]


def residue_uid(resid, icode=""):
    icode = str(icode or "").strip()
    return f"{int(resid)}{icode}" if icode else str(int(resid))


def raw_residue_records(path):
    """Return receptor residues in the same raw PDB order used during label construction."""
    structure = read_pdb(str(path))
    seen = set()
    residues = []
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
        residues.append(
            {
                "key": key,
                "uid": residue_uid(resid, icode),
                "aa": res3to1.get(str(structure["resname"][i]).strip().upper(), "X"),
            }
        )
    return residues


def cleaned_residue_sequence_and_count(path):
    """Return residue sequence after PeSTo clean/encode, plus encoded residue count."""
    structure = clean_structure(read_pdb(str(path)))
    seen = set()
    seq = []
    for resid, resname in zip(structure["resid"].tolist(), structure["resname"].tolist()):
        resid = int(resid)
        if resid in seen:
            continue
        seen.add(resid)
        seq.append(res3to1.get(str(resname).strip().upper(), "X"))
    _x, m = encode_structure(structure)
    return seq, int(m.shape[1])


def check_row(payload):
    index, row = payload
    case_id = row.get("apex_case_id", "")
    labels = split_field(row.get("labels", ""))
    mask = split_field(row.get("label_mask", ""))
    manifest_ids = split_field(row.get("input_residue_ids", ""))
    try:
        raw_records = raw_residue_records(row["receptor_pdb_path"])
        raw_ids = [r["uid"] for r in raw_records]
        raw_seq = [r["aa"] for r in raw_records]
        clean_seq, encoded_residues = cleaned_residue_sequence_and_count(row["receptor_pdb_path"])
    except Exception as exc:
        return {
            "index": index,
            "ok": False,
            "reason": f"structure_error:{type(exc).__name__}:{str(exc)[:160]}",
            "case_id": case_id,
            "labels_len": len(labels),
            "mask_len": len(mask),
            "manifest_len": len(manifest_ids),
            "encoded_residues": "",
        }

    lengths_ok = len(labels) == len(mask) == len(manifest_ids) == len(raw_ids) == encoded_residues
    if not lengths_ok:
        reason = "ordinal_len_mismatch"
        ok = False
    elif manifest_ids != raw_ids:
        reason = "manifest_not_raw_pdb_order"
        ok = False
    elif raw_seq != clean_seq:
        reason = "clean_changed_residue_sequence_order"
        ok = False
    else:
        reason = "ordinal_alignment_ok"
        ok = True

    return {
        "index": index,
        "ok": ok,
        "reason": reason,
        "case_id": case_id,
        "labels_len": len(labels),
        "mask_len": len(mask),
        "manifest_len": len(manifest_ids),
        "raw_len": len(raw_ids),
        "encoded_residues": encoded_residues,
        "manifest_head": FIELD_SEP.join(manifest_ids[:12]),
        "raw_head": FIELD_SEP.join(raw_ids[:12]),
        "manifest_tail": FIELD_SEP.join(manifest_ids[-8:]),
        "raw_tail": FIELD_SEP.join(raw_ids[-8:]),
    }


def read_rows(path, limit=0):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[: int(limit)]
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Audit APEX manifest rows for ordinal residue-label alignment. "
            "This checks whether labels[i] follows the same raw PDB residue order "
            "that PeSTo clean/encode preserves after renumbering."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(input_path, args.limit)
    if not rows:
        raise RuntimeError(f"No rows found in {input_path}")

    start = time.time()
    results = []
    workers = max(int(args.workers), 1)
    iterator = enumerate(rows)
    if workers == 1:
        mapped = map(check_row, iterator)
        for completed, result in enumerate(mapped, 1):
            results.append(result)
            if completed == 1 or completed % args.progress_every == 0:
                print(f"checked {completed}/{len(rows)} elapsed={time.time() - start:.1f}s", flush=True)
    else:
        with Pool(processes=workers) as pool:
            for completed, result in enumerate(pool.imap_unordered(check_row, iterator, chunksize=8), 1):
                results.append(result)
                if completed == 1 or completed % args.progress_every == 0:
                    print(f"checked {completed}/{len(rows)} elapsed={time.time() - start:.1f}s", flush=True)

    results.sort(key=lambda x: x["index"])
    kept = [rows[r["index"]] for r in results if r["ok"]]
    dropped = []
    for r in results:
        if r["ok"]:
            continue
        row = dict(rows[r["index"]])
        for key, value in r.items():
            if key != "index":
                row[f"_ordinal_{key}"] = value
        dropped.append(row)

    fieldnames = list(rows[0].keys())
    dropped_fieldnames = fieldnames + [
        "_ordinal_ok",
        "_ordinal_reason",
        "_ordinal_case_id",
        "_ordinal_labels_len",
        "_ordinal_mask_len",
        "_ordinal_manifest_len",
        "_ordinal_raw_len",
        "_ordinal_encoded_residues",
        "_ordinal_manifest_head",
        "_ordinal_raw_head",
        "_ordinal_manifest_tail",
        "_ordinal_raw_tail",
    ]
    write_csv(output_dir / "apex_paired_manifest.csv", kept, fieldnames)
    write_csv(output_dir / "dropped_ordinal_alignment_rows.csv", dropped, dropped_fieldnames)

    reason_counts = Counter(r["reason"] for r in results)
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_manifest": str(input_path),
        "output_manifest": str(output_dir / "apex_paired_manifest.csv"),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(dropped),
        "reason_counts": dict(reason_counts),
        "elapsed_seconds": round(time.time() - start, 2),
        "definition": (
            "A row is kept when labels/mask/input_residue_ids/raw-PDB residues/PeSTo encoded "
            "residue count have equal length, manifest input_residue_ids equal raw PDB residue "
            "order, and PeSTo clean_structure preserves residue sequence order while only "
            "renumbering residues."
        ),
    }
    (output_dir / "ordinal_alignment_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
