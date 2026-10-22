import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description='Run backbone examples from an isolated application copy')
    parser.add_argument('--dependency-path', type=Path, help='Optional installed third-party package directory, not application source')
    arguments = parser.parse_args()
    dependency = arguments.dependency_path.resolve() if arguments.dependency_path else None
    if dependency and any((dependency / name).exists() for name in ['gsbind', 'gaussian_direct']):
        raise ValueError('Dependency path must not contain application source')
    with tempfile.TemporaryDirectory(prefix='gsbind_portable_') as temporary:
        destination = Path(temporary) / 'GSbind'
        destination.mkdir()
        for name in ['src', 'configs', 'examples']:
            shutil.copytree(ROOT / name, destination / name, ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2(ROOT / 'main.py', destination / 'main.py')
        (destination / 'checkpoints').mkdir()
        shutil.copy2(ROOT / 'checkpoints/direct128_e11.pt', destination / 'checkpoints/direct128_e11.pt')
        cases = pd.read_csv(ROOT / 'examples/cases.csv')
        for column in ['compact_path', 'receptor_pdb_path']:
            for value in cases[column]:
                target = destination / value
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / value, target)
        pythonpath = str(destination / 'src') + (os.pathsep + str(dependency) if dependency else '')
        environment = dict(os.environ, PYTHONPATH=pythonpath, OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2')
        result = subprocess.run([sys.executable, str(destination / 'main.py'), '--config', str(destination / 'configs/smoke.json'), 'evaluate'],
                                cwd=temporary, env=environment, capture_output=True, text=True)
        (ROOT / 'provenance/portable_smoke.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr)
        metrics = json.loads((destination / 'runs/example_metrics.json').read_text())
        verification = dict(status='passed', standalone_directory=True, repository_pythonpath=False, cases=metrics['num_cases'],
                            pooled_mcc=metrics['pooled_binary_metrics']['mcc'])
        (ROOT / 'provenance/portable_smoke.json').write_text(json.dumps(verification, indent=2) + '\n')
        print(json.dumps(verification))


if __name__ == '__main__':
    main()
