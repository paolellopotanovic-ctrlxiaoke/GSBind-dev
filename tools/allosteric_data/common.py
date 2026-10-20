import hashlib
import json
import os
from pathlib import Path
import sys
import time

import requests


PACKAGE = Path(__file__).resolve().parents[2]
ROOT = PACKAGE / 'datasets/allosteric'
LEGACY = ROOT / 'reproduction_cache'
SUPPLIED = ROOT / 'raw/supplied'
sys.path.insert(0, str(PACKAGE / 'src'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def fetch(url, path, retries=3):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=(12, 40))
            response.raise_for_status()
            temporary = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
            temporary.write_bytes(response.content)
            temporary.replace(path)
            return response.content
        except requests.RequestException:
            if attempt + 1 == retries:
                raise
            time.sleep(2 * (attempt + 1))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
