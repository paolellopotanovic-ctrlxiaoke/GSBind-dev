import argparse
import csv
from pathlib import Path


def residue_count(row: dict[str, str]) -> int:
    for key in ("label_mask", "labels"):
        value = row.get(key, "")
        if value:
            return len(value.split(";"))
    return 10**9


def main() -> None:
    parser = argparse.ArgumentParser(description="Select the shortest cached regression case.")
    parser.add_argument("manifest")
    parser.add_argument("output")
    args = parser.parse_args()
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not rows or not fieldnames:
        raise SystemExit("manifest is empty")
    selected = min(rows, key=residue_count)
    with Path(args.output).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(selected)
    print(selected.get("apex_case_id", selected.get("case_id", "unknown")), residue_count(selected))


if __name__ == "__main__":
    main()
