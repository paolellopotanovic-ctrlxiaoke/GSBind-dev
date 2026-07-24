#!/usr/bin/env python3
"""Build large APEX paired manifests from Pinder files without PinderSystem.

The remote runtime currently has a 2GB cgroup memory limit, while
`pinder.core.PinderSystem` can be killed during initialization. This script
therefore uses only lightweight pieces:

- `pyarrow.dataset` for selecting rows from `index.parquet`
- direct HTTPS downloads from `https://storage.googleapis.com/pinder/2024-02`
- M4's `m4.data.structure_io.read_pdb` for atom/residue parsing
- Pinder mapping parquet files for residue -> UniProt transfer
"""

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable, **_kwargs):
        return iterable


ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

from m4.data.structure_io import read_pdb


FIELD_SEP = ";"
AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}
PINDER_INDEX_METADATA_FIELDS = [
    "split",
    "cluster_id",
    "cluster_id_R",
    "cluster_id_L",
    "uniprot_R",
    "uniprot_L",
    "pinder_s",
    "pinder_xl",
    "pinder_af2",
    "apo_R_quality",
    "apo_L_quality",
]


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y"}
    return bool(value)


def pinder_index_value(row, key):
    value = row.get(key, "")
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        pass
    return str(value)


def sanitize_id(value):
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))


def opposite_side(side):
    return "L" if side == "R" else "R"


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt" if str(path).endswith(".gz") else "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def download_file(base_url, prefix, name, dest, retries=3, timeout=120):
    if not name:
        raise FileNotFoundError("empty Pinder filename")
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/{prefix}/{name}"
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.tmp")
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ["curl", "-L", "--fail", "--silent", "--show-error", "--max-time", str(timeout), "-o", str(tmp), url],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0 and tmp.is_file() and tmp.stat().st_size > 0:
            if not dest.exists():
                tmp.replace(dest)
            else:
                tmp.unlink()
            return dest
        if tmp.exists():
            tmp.unlink()
        if attempt == retries:
            raise RuntimeError(f"download failed: {url}: {result.stderr.strip()}")
    return dest


def ensure_pdb(args, name):
    root = Path(args.pinder_root)
    return download_file(
        args.base_url,
        "pdbs",
        name,
        root / "pdbs" / name,
        retries=args.download_retries,
        timeout=args.download_timeout,
    )


def ensure_mapping(args, pdb_name):
    root = Path(args.pinder_root)
    mapping_name = Path(pdb_name).with_suffix(".parquet").name
    return download_file(
        args.base_url,
        "mappings",
        mapping_name,
        root / "mappings" / mapping_name,
        retries=args.download_retries,
        timeout=args.download_timeout,
    )


def residue_uid(resid, icode=""):
    icode = str(icode or "").strip()
    return f"{int(resid)}{icode}" if icode else str(int(resid))


def residue_records_from_pdb(path):
    structure = read_pdb(str(path))
    seen = set()
    residues = []
    for i, resid in enumerate(structure["resid"]):
        if str(structure["het_flag"][i]) != "A":
            continue
        chain = str(structure["chain_name"][i]).split(":")[0]
        icode = str(structure.get("icode", [""])[i]).strip()
        key = (chain, int(resid), icode)
        if key in seen:
            continue
        seen.add(key)
        residues.append(
            {
                "key": key,
                "chain": chain,
                "resid": int(resid),
                "icode": icode,
                "uid": residue_uid(resid, icode),
                "resname": str(structure["resname"][i]).strip().upper(),
            }
        )
    return residues, structure


def sequence_from_pdb(path):
    residues, _structure = residue_records_from_pdb(path)
    return "".join(AA3_TO_AA1.get(res["resname"], "X") for res in residues)


def heavy_atoms_by_side(native_path):
    structure = read_pdb(str(native_path))
    atoms = {"R": [], "L": []}
    for i, xyz in enumerate(structure["xyz"]):
        chain = str(structure["chain_name"][i]).split(":")[0]
        if chain not in atoms:
            continue
        atom_name = str(structure["name"][i]).strip().upper()
        element = str(structure["element"][i]).strip().upper()
        if element == "H" or atom_name.startswith("H"):
            continue
        if str(structure["het_flag"][i]) != "A":
            continue
        icode = str(structure.get("icode", [""])[i]).strip()
        key = (chain, int(structure["resid"][i]), icode)
        atoms[chain].append((key, np.asarray(xyz, dtype=np.float32)))
    return atoms


def compute_native_interface_from_pdb(native_path, cutoff):
    atoms = heavy_atoms_by_side(native_path)
    r_atoms = atoms["R"]
    l_atoms = atoms["L"]
    if not r_atoms or not l_atoms:
        raise ValueError("native dimer does not contain both R/L heavy atoms")
    r_keys, r_xyz = zip(*r_atoms)
    l_keys, l_xyz = zip(*l_atoms)
    r_xyz = np.asarray(r_xyz, dtype=np.float32)
    l_xyz = np.asarray(l_xyz, dtype=np.float32)
    cutoff2 = cutoff * cutoff
    positives = {"R": set(), "L": set()}
    for start in range(0, len(r_xyz), 512):
        d = r_xyz[start : start + 512, None, :] - l_xyz[None, :, :]
        hits = np.argwhere(np.sum(d * d, axis=2) < cutoff2)
        for local_i, j in hits:
            positives["R"].add(r_keys[start + int(local_i)])
            positives["L"].add(l_keys[int(j)])
    return positives


def mapping_from_parquet(mapping_path):
    table = pq.read_table(mapping_path, columns=["resi_pdb", "resi_auth", "resi_uniprot", "uniprot_acc", "resolved"])
    df = table.to_pandas()
    mapping = {}
    for _idx, row in df.iterrows():
        uni = row.get("resi_uniprot")
        if uni is None or str(uni) == "<NA>" or str(uni) == "nan":
            continue
        candidates = []
        for key in ("resi_pdb", "resi_auth"):
            value = row.get(key)
            if value is None or str(value) == "nan" or str(value) == "<NA>":
                continue
            try:
                candidates.append(int(float(value)))
            except ValueError:
                continue
        for resid in candidates:
            mapping[resid] = int(uni)
    return mapping


def build_labels(input_pdb, input_mapping_path, positive_uniprots, native_positive_resids):
    residues, _structure = residue_records_from_pdb(input_pdb)
    mapping = mapping_from_parquet(input_mapping_path)
    labels = []
    mask = []
    mapped_positive = []
    mapped_uniprots = []
    for res in residues:
        uni = mapping.get(res["resid"])
        mapped_uniprots.append("" if uni is None else str(uni))
        if uni is not None:
            y = uni in positive_uniprots
            valid = True
        else:
            y = res["resid"] in native_positive_resids
            valid = False
        labels.append(int(y))
        mask.append(int(valid))
        if y:
            mapped_positive.append(res["uid"])
    return residues, labels, mask, mapped_positive, mapped_uniprots


def iter_index_rows(index_path, split_name, states, batch_size, allowed_system_ids=None):
    dataset = ds.dataset(index_path, format="parquet")
    state_filters = []
    if "apo" in states:
        state_filters.append((ds.field("apo_R") == True) | (ds.field("apo_L") == True))
    if "af2" in states:
        state_filters.append((ds.field("predicted_R") == True) | (ds.field("predicted_L") == True))
    filter_expr = ds.field("split") == split_name
    if state_filters:
        state_expr = state_filters[0]
        for extra in state_filters[1:]:
            state_expr = state_expr | extra
        filter_expr = filter_expr & state_expr
    elif "holo" not in states:
        raise ValueError("At least one of holo/apo/af2 must be requested.")
    if allowed_system_ids:
        filter_expr = filter_expr & ds.field("id").isin(sorted(allowed_system_ids))
    columns = [
        "id",
        "split",
        "pdb_id",
        "cluster_id",
        "cluster_id_R",
        "cluster_id_L",
        "pinder_s",
        "pinder_xl",
        "pinder_af2",
        "uniprot_R",
        "uniprot_L",
        "holo_R_pdb",
        "holo_L_pdb",
        "predicted_R_pdb",
        "predicted_L_pdb",
        "apo_R_pdb",
        "apo_L_pdb",
        "predicted_R",
        "predicted_L",
        "apo_R",
        "apo_L",
        "apo_R_quality",
        "apo_L_quality",
    ]
    scanner = dataset.scanner(filter=filter_expr, columns=columns, batch_size=batch_size)
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            yield row


def desired_cases(row, states):
    cases = []
    for state in states:
        for side in ("R", "L"):
            if state == "holo" and row.get(f"holo_{side}_pdb", ""):
                cases.append((state, side, row.get(f"holo_{side}_pdb", "")))
            if state == "apo" and as_bool(row.get(f"apo_{side}", False)):
                cases.append((state, side, row.get(f"apo_{side}_pdb", "")))
            if state == "af2" and as_bool(row.get(f"predicted_{side}", False)):
                cases.append((state, side, row.get(f"predicted_{side}_pdb", "")))
    return cases


def build_manifest(args):
    states = [x.strip() for x in args.states.split(",") if x.strip()]
    allowed_cases = None
    allowed_system_ids = None
    if args.case_index:
        allowed_cases = set()
        for row in read_csv(args.case_index):
            system_id = str(row.get("system_id", "")).strip()
            side = str(row.get("side") or row.get("receptor_side") or "").strip().upper()
            apo_name = Path(
                str(
                    row.get("apo_pdb_basename")
                    or row.get("source_receptor_pdb")
                    or row.get("receptor_pdb_path")
                    or ""
                )
            ).name
            if system_id and side in {"R", "L"} and apo_name:
                allowed_cases.add((system_id, side, apo_name))
        if not allowed_cases:
            raise ValueError(f"No valid cases found in --case-index={args.case_index}")
        allowed_system_ids = {case[0] for case in allowed_cases}
    index_iter = iter_index_rows(
        args.index_path,
        args.split,
        states,
        args.index_batch_size,
        allowed_system_ids=allowed_system_ids,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_pdb_root = output_dir / "pdbs"
    partial_manifest = output_dir / "apex_paired_manifest.partial.csv"
    partial_errors = output_dir / "apex_paired_manifest_errors.partial.jsonl"
    progress_path = output_dir / "apex_paired_manifest_progress.json"
    manifest_rows = []
    errors = []
    skip_reasons = Counter()
    resume_from = 0
    resume_shard_processed = 0
    if progress_path.exists() and partial_manifest.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        resume_from = int(progress.get("processed_systems", 0))
        resume_shard_processed = int(progress.get("shard_processed", 0) or 0)
        skip_reasons.update(progress.get("skip_reasons", {}))
        manifest_rows = read_csv(partial_manifest)
        errors = read_jsonl(partial_errors)
        print(f"Resuming after {resume_from} scanned systems, rows={len(manifest_rows)}, errors={len(errors)}", flush=True)

    def flush(processed):
        write_csv(partial_manifest, manifest_rows)
        write_jsonl(partial_errors, errors)
        progress = {
            "processed_systems": processed,
            "shard_processed": resume_shard_processed + shard_processed,
            "total_systems": "",
            "manifest_rows": len(manifest_rows),
            "errors": len(errors),
            "skip_reasons": dict(sorted(skip_reasons.items())),
            "target_cases": args.target_cases,
            "states": states,
            "output_dir": str(output_dir),
        }
        progress_path.write_text(json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8")

    last_processed = resume_from
    scanned = 0
    shard_processed = 0
    for row in tqdm(index_iter, desc=f"apex_file_pinder_s{args.shard_id:02d}"):
        scanned += 1
        if scanned <= resume_from:
            continue
        if args.limit_systems and scanned > args.limit_systems:
            break
        if args.num_shards > 1 and ((scanned - 1) % args.num_shards) != args.shard_id:
            continue
        shard_processed += 1
        offset = scanned
        last_processed = offset
        if args.target_cases and len(manifest_rows) >= args.target_cases:
            break
        system_id = str(row["id"])
        safe_system = sanitize_id(system_id)
        system_stage = "start"
        try:
            native_name = f"{system_id}.pdb"
            system_stage = "native_pdb"
            native_pdb = ensure_pdb(args, native_name)
            system_stage = "holo_R_pdb"
            holo_r_pdb = ensure_pdb(args, row.get("holo_R_pdb", ""))
            system_stage = "holo_L_pdb"
            holo_l_pdb = ensure_pdb(args, row.get("holo_L_pdb", ""))
            holo_pdbs = {
                "R": holo_r_pdb,
                "L": holo_l_pdb,
            }
            system_stage = "holo_R_mapping"
            holo_r_map = ensure_mapping(args, row.get("holo_R_pdb", ""))
            system_stage = "holo_L_mapping"
            holo_l_map = ensure_mapping(args, row.get("holo_L_pdb", ""))
            holo_maps = {
                "R": holo_r_map,
                "L": holo_l_map,
            }
            system_stage = "native_interface"
            native_interface = compute_native_interface_from_pdb(native_pdb, args.contact_cutoff)
            if not native_interface["R"] and not native_interface["L"]:
                skip_reasons["system:no_native_contacts"] += 1
                if shard_processed % args.flush_every == 0:
                    flush(offset)
                continue
            positive_uniprots = {}
            for side in ("R", "L"):
                system_stage = f"holo_{side}_mapping_parse"
                mapping = mapping_from_parquet(holo_maps[side])
                positive_uniprots[side] = {mapping[key[1]] for key in native_interface[side] if key[1] in mapping}
        except Exception as exc:
            skip_reasons[f"system:{system_stage}_error"] += 1
            errors.append({"system_id": system_id, "status": "system_error", "stage": system_stage, "error": repr(exc)})
            if shard_processed % args.flush_every == 0:
                flush(offset)
            continue

        apo_pair = as_bool(row.get("apo_R", False)) and as_bool(row.get("apo_L", False))
        for state, receptor_side, receptor_name in desired_cases(row, states):
            if allowed_cases is not None and (system_id, receptor_side, Path(str(receptor_name)).name) not in allowed_cases:
                continue
            if args.target_cases and len(manifest_rows) >= args.target_cases:
                break
            partner_side = opposite_side(receptor_side)
            case_stage = "start"
            try:
                case_stage = "receptor_pdb"
                receptor_pdb = ensure_pdb(args, receptor_name)
                case_stage = "receptor_mapping"
                receptor_map = ensure_mapping(args, receptor_name)
                case_stage = "partner_sequence"
                partner_seq = sequence_from_pdb(holo_pdbs[partner_side])
                if not partner_seq:
                    skip_reasons["case:empty_partner_sequence"] += 1
                    continue
                case_stage = "labels"
                residues, labels, mask, mapped_positive, mapped_uniprots = build_labels(
                    receptor_pdb,
                    receptor_map,
                    positive_uniprots[receptor_side],
                    {key[1] for key in native_interface[receptor_side]},
                )
                if args.max_residues and len(residues) > args.max_residues:
                    skip_reasons["case:max_residues"] += 1
                    continue
                num_pos = int(sum(1 for y, m in zip(labels, mask) if y and m))
                if args.require_positive and num_pos <= 0:
                    if not positive_uniprots[receptor_side]:
                        skip_reasons["case:no_positive_labels_empty_positive_uniprots"] += 1
                    elif sum(labels) > 0 and sum(mask) == 0:
                        skip_reasons["case:no_positive_labels_unmasked_only"] += 1
                    elif sum(labels) > 0:
                        skip_reasons["case:no_positive_labels_masked_out"] += 1
                    else:
                        skip_reasons["case:no_positive_labels_no_label_hits"] += 1
                    continue
                mask_coverage = float(sum(mask)) / float(max(len(residues), 1))
                if mask_coverage < args.min_mapping_coverage and not args.keep_low_mapping_quality:
                    skip_reasons["case:low_mapping_coverage"] += 1
                    continue
                skip_reasons["case:success"] += 1
                out_receptor = Path(receptor_pdb)
                if args.copy_receptor_pdb:
                    out_receptor = out_pdb_root / safe_system / f"{safe_system}_{state}_{receptor_side}.pdb"
                    out_receptor.parent.mkdir(parents=True, exist_ok=True)
                    if not out_receptor.exists():
                        out_receptor.write_bytes(Path(receptor_pdb).read_bytes())
                manifest_rows.append(
                    {
                        "apex_case_id": f"{safe_system}:{state}:{receptor_side}|partner:{partner_side}",
                        "receptor_case_id": f"{safe_system}:{state}:{receptor_side}",
                        "partner_case_id": f"{safe_system}:holo:{partner_side}",
                        "system_id": system_id,
                        "pdb_id": str(row.get("pdb_id", "")).lower(),
                        "benchmark": args.benchmark_name,
                        **{field: pinder_index_value(row, field) for field in PINDER_INDEX_METADATA_FIELDS},
                        "input_state": state,
                        "receptor_side": receptor_side,
                        "partner_side": partner_side,
                        "receptor_pdb_path": str(out_receptor),
                        "partner_pdb_path": str(holo_pdbs[partner_side]),
                        "partner_sequence": partner_seq,
                        "partner_length": len(partner_seq),
                        "native_interface_residue_ids": FIELD_SEP.join(
                            sorted([residue_uid(k[1], k[2]) for k in native_interface[receptor_side]], key=str)
                        ),
                        "mapped_input_residue_ids": FIELD_SEP.join(sorted(mapped_positive, key=str)),
                        "label_mask": FIELD_SEP.join(str(int(x)) for x in mask),
                        "labels": FIELD_SEP.join(str(int(x)) for x in labels),
                        "input_residue_ids": FIELD_SEP.join(res["uid"] for res in residues),
                        "mapped_uniprot_positions": FIELD_SEP.join(mapped_uniprots),
                        "num_residues": len(residues),
                        "num_positive_labels": num_pos,
                        "mask_coverage": f"{mask_coverage:.6f}",
                        "low_mapping_quality": int(mask_coverage < args.min_mapping_coverage),
                        "apo_pair_system": int(apo_pair),
                        "source_receptor_pdb": receptor_name,
                        "source_partner_pdb": Path(holo_pdbs[partner_side]).name,
                        "mapping_method": "pinder_mapping_parquet",
                        "label_source": "native_holo_heavy_atom_contact_5A_transferred_by_pinder_mapping",
                    }
                )
            except Exception as exc:
                skip_reasons[f"case:{case_stage}_error"] += 1
                errors.append(
                    {
                        "system_id": system_id,
                        "input_state": state,
                        "receptor_side": receptor_side,
                        "partner_side": partner_side,
                        "status": "case_error",
                        "stage": case_stage,
                        "error": repr(exc),
                    }
                )
        if shard_processed % args.flush_every == 0:
            flush(offset)
    write_csv(output_dir / "apex_paired_manifest.csv", manifest_rows)
    write_jsonl(output_dir / "apex_paired_manifest_errors.jsonl", errors)
    flush(last_processed)
    counts = Counter((row["input_state"], row["receptor_side"]) for row in manifest_rows)
    summary = {
        "rows": len(manifest_rows),
        "systems_processed": last_processed,
        "shard_processed": resume_shard_processed + shard_processed,
        "candidate_systems": "",
        "errors": len(errors),
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "states": states,
        "counts": {f"{state}:{side}": count for (state, side), count in sorted(counts.items())},
        "output_dir": str(output_dir),
    }
    (output_dir / "apex_paired_manifest_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build large paired APEX manifest from Pinder files.")
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--pinder-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--states", default="holo")
    parser.add_argument(
        "--case-index",
        default="",
        help="Optional CSV whitelist with system_id, side/receptor_side and apo_pdb_basename.",
    )
    parser.add_argument("--target-cases", type=int, default=100000)
    parser.add_argument("--limit-systems", type=int, default=0)
    parser.add_argument("--index-batch-size", type=int, default=512)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--benchmark-name", default="pinder_apex_train_large")
    parser.add_argument("--base-url", default="https://storage.googleapis.com/pinder/2024-02")
    parser.add_argument("--download-retries", type=int, default=3)
    parser.add_argument("--download-timeout", type=int, default=120)
    parser.add_argument("--contact-cutoff", type=float, default=5.0)
    parser.add_argument("--min-mapping-coverage", type=float, default=0.70)
    parser.add_argument("--max-residues", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=50)
    parser.add_argument("--require-positive", action="store_true", default=True)
    parser.add_argument("--keep-low-mapping-quality", action="store_true")
    parser.add_argument("--copy-receptor-pdb", action="store_true")
    args = parser.parse_args()
    build_manifest(args)


if __name__ == "__main__":
    main()
