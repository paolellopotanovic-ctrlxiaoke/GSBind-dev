import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from m4.cli import inference_main


if __name__ == "__main__":
    inference_main()
