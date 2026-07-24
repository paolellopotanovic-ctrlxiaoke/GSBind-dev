import argparse
import csv
import json
import os
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from ..data.manifest import build_tensor_item, cache_path_for_row, save_tensor_item


def configure_torch_threads(num_threads):
    try:
        import torch as pt

        pt.set_num_threads(max(1, int(num_threads)))
        pt.set_num_interop_threads(1)
    except Exception:
        pass


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def worker(payload):
    index, row, cache_dir, max_num_nn, strict_alignment, skip_existing, trust_final_filter = payload
    cache_path = cache_path_for_row(cache_dir, row)
    case_id = row.get("apex_case_id", "")
    if skip_existing and cache_path.is_file() and cache_path.stat().st_size > 0:
        return {
            "index": index,
            "ok": True,
            "status": "exists",
            "case_id": case_id,
            "cache_path": str(cache_path),
            "num_residues": int(row.get("num_residues") or 0),
        }
    try:
        item = build_tensor_item(
            row,
            max_num_nn=max_num_nn,
            strict_residue_alignment=(strict_alignment and not trust_final_filter),
            skip_bad_examples=False,
        )
        if trust_final_filter:
            labels_len = len(str(row.get("labels", "")).split(";")) if row.get("labels", "") else 0
            mask_len = len(str(row.get("label_mask", "")).split(";")) if row.get("label_mask", "") else 0
            encoded_len = int(item["M"].shape[1])
            expected_len = int(row.get("num_residues") or encoded_len)
            if labels_len != encoded_len or mask_len != encoded_len or expected_len != encoded_len:
                return {
                    "index": index,
                    "ok": False,
                    "status": "error",
                    "case_id": case_id,
                    "reason": (
                        "trusted_final_filter_length_mismatch:"
                        f"labels={labels_len},mask={mask_len},num_residues={expected_len},encoded={encoded_len}"
                    ),
                }
        if item.get("skip"):
            return {
                "index": index,
                "ok": False,
                "status": "skip",
                "case_id": case_id,
                "reason": item.get("skip_reason", "skip"),
            }
        save_tensor_item(cache_path, item)
        return {
            "index": index,
            "ok": True,
            "status": "written",
            "case_id": case_id,
            "cache_path": str(cache_path),
            "num_residues": int(item["M"].shape[1]),
            "num_atoms": int(item["X"].shape[0]),
        }
    except Exception as exc:
        return {
            "index": index,
            "ok": False,
            "status": "error",
            "case_id": case_id,
            "reason": f"{type(exc).__name__}:{str(exc)[:200]}",
        }


def main():
    parser = argparse.ArgumentParser(description="Precompute M4 per-row tensor caches from a manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-num-nn", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Intra-op torch threads per worker. Keep this at 1 for many CPU workers.",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-strict-residue-alignment", action="store_true")
    parser.add_argument(
        "--trust-final-filter",
        action="store_true",
        help=(
            "Skip the expensive raw-vs-clean ordinal residue audit because the input manifest "
            "already passed the final manifest audit. Length checks are still enforced."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(manifest)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError(f"No rows found in {manifest}")

    configure_torch_threads(args.torch_threads)
    start = time.time()
    results = []
    tasks = [
        (
            index,
            row,
            str(cache_dir),
            args.max_num_nn,
            not args.no_strict_residue_alignment,
            not args.overwrite,
            args.trust_final_filter,
        )
        for index, row in enumerate(rows)
    ]
    workers = max(int(args.workers), 1)
    if workers == 1:
        iterator = map(worker, tasks)
        for completed, result in enumerate(iterator, 1):
            results.append(result)
            if completed == 1 or completed % args.progress_every == 0:
                elapsed = time.time() - start
                print(f"cached {completed}/{len(tasks)} elapsed={elapsed:.1f}s", flush=True)
    else:
        with Pool(processes=workers) as pool:
            for completed, result in enumerate(pool.imap_unordered(worker, tasks, chunksize=8), 1):
                results.append(result)
                if completed == 1 or completed % args.progress_every == 0:
                    elapsed = time.time() - start
                    print(f"cached {completed}/{len(tasks)} elapsed={elapsed:.1f}s", flush=True)

    status_counts = Counter(r.get("status", "") for r in results)
    error_reasons = Counter(r.get("reason", "") for r in results if not r.get("ok"))
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(manifest),
        "cache_dir": str(cache_dir),
        "input_rows": len(rows),
        "ok_rows": sum(1 for r in results if r.get("ok")),
        "bad_rows": sum(1 for r in results if not r.get("ok")),
        "status_counts": dict(status_counts),
        "top_error_reasons": dict(error_reasons.most_common(20)),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    (cache_dir / "tensor_cache_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    failures = [r for r in sorted(results, key=lambda x: x["index"]) if not r.get("ok")]
    if failures:
        with (cache_dir / "tensor_cache_failures.jsonl").open("w", encoding="utf-8") as handle:
            for item in failures:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
