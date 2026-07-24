import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import torch as pt


from ..data.manifest import build_tensor_item, cache_key_for_row, cache_path_for_row


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def graph_cache_key(row, ks):
    raw = "|".join([cache_key_for_row(row), ",".join(str(k) for k in ks)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20] + ".pt"


def residue_centers(X, M):
    weights = M.to(dtype=X.dtype)
    denom = weights.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
    return weights.transpose(0, 1).matmul(X) / denom


def build_graph_item(item, ks):
    X = item["X"].float()
    M = item["M"].float()
    centers = residue_centers(X, M)
    n = int(centers.shape[0])
    out = {
        "centers": centers.half(),
        "num_residues": n,
        "ks": list(ks),
    }
    if n <= 1:
        for k in ks:
            out[f"knn_idx_k{k}"] = pt.empty((n, 0), dtype=pt.long)
            out[f"knn_dist_k{k}"] = pt.empty((n, 0), dtype=pt.float16)
        return out
    dist = pt.cdist(centers, centers).clamp_min(0.0)
    for k in ks:
        kk = min(int(k) + 1, n)
        nn_dist, nn_idx = pt.topk(dist, k=kk, dim=-1, largest=False)
        out[f"knn_idx_k{k}"] = nn_idx[:, 1:].contiguous().long()
        out[f"knn_dist_k{k}"] = nn_dist[:, 1:].contiguous().half()
    return out


def worker(payload):
    index, row, tensor_cache_dir, graph_cache_dir, ks, max_num_nn, strict_alignment, skip_existing = payload
    out_path = Path(graph_cache_dir) / graph_cache_key(row, ks)
    case_id = row.get("apex_case_id", "")
    if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
        return {"index": index, "ok": True, "status": "exists", "case_id": case_id, "path": str(out_path)}
    try:
        tensor_path = cache_path_for_row(tensor_cache_dir, row) if tensor_cache_dir else None
        if tensor_path and tensor_path.is_file():
            item = pt.load(tensor_path, map_location="cpu")
        else:
            item = build_tensor_item(
                row,
                max_num_nn=max_num_nn,
                strict_residue_alignment=strict_alignment,
                skip_bad_examples=False,
            )
        graph = build_graph_item(item, ks=ks)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pt.save(graph, out_path)
        return {
            "index": index,
            "ok": True,
            "status": "written",
            "case_id": case_id,
            "path": str(out_path),
            "num_residues": int(graph["num_residues"]),
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
    parser = argparse.ArgumentParser(description="Precompute residue-center kNN graph caches for M4.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--graph-cache-dir", required=True)
    parser.add_argument("--tensor-cache-dir", default="")
    parser.add_argument("--ks", default="16,24,32")
    parser.add_argument("--max-num-nn", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-strict-residue-alignment", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = read_rows(args.manifest)
    if args.limit:
        rows = rows[: args.limit]
    ks = sorted({int(x) for x in args.ks.split(",") if x.strip()})
    out_dir = Path(args.graph_cache_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    tasks = [
        (
            i,
            row,
            args.tensor_cache_dir,
            str(out_dir),
            ks,
            args.max_num_nn,
            not args.no_strict_residue_alignment,
            not args.overwrite,
        )
        for i, row in enumerate(rows)
    ]
    results = []
    workers = max(1, int(args.workers))
    if workers == 1:
        iterator = map(worker, tasks)
    else:
        pool = Pool(processes=workers)
        iterator = pool.imap_unordered(worker, tasks, chunksize=16)
    try:
        for completed, result in enumerate(iterator, 1):
            results.append(result)
            if completed == 1 or completed % args.progress_every == 0:
                elapsed = time.time() - start
                print(f"graph_cached {completed}/{len(tasks)} elapsed={elapsed:.1f}s", flush=True)
    finally:
        if workers != 1:
            pool.close()
            pool.join()

    status_counts = Counter(r.get("status", "") for r in results)
    errors = Counter(r.get("reason", "") for r in results if not r.get("ok"))
    report = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(Path(args.manifest)),
        "graph_cache_dir": str(out_dir),
        "tensor_cache_dir": args.tensor_cache_dir,
        "input_rows": len(rows),
        "ok_rows": sum(1 for r in results if r.get("ok")),
        "bad_rows": sum(1 for r in results if not r.get("ok")),
        "ks": ks,
        "status_counts": dict(status_counts),
        "top_error_reasons": dict(errors.most_common(20)),
        "elapsed_seconds": round(time.time() - start, 2),
    }
    (out_dir / "residue_graph_cache_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if errors:
        with (out_dir / "residue_graph_cache_failures.jsonl").open("w", encoding="utf-8") as handle:
            for item in sorted(results, key=lambda x: x["index"]):
                if not item.get("ok"):
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
