import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from utils.reproducibility import verify_validated_sources


def main() -> None:
    results = verify_validated_sources(ROOT)
    failed = False
    for name, (expected, actual) in results.items():
        status = "OK" if expected == actual else "MISMATCH"
        print(f"{status:8} {name} {actual}")
        failed |= expected != actual
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
