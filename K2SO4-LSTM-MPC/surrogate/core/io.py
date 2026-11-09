"""Atomic result and checkpoint writes for resuming the same run."""
import csv
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_file(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    writer(tmp)
    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.1*(attempt+1))


def write_json(path, payload):
    atomic_file(path, lambda p:p.write_text(json.dumps(payload, ensure_ascii=False,
                indent=2, allow_nan=False), encoding='utf-8'))


def save_npz(path, **arrays):
    def writer(p):
        with p.open('wb') as f:
            np.savez_compressed(f, **arrays)
    atomic_file(path, writer)


def save_checkpoint(path, payload):
    atomic_file(path, lambda p:torch.save(payload, p))


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    def writer(p):
        with p.open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    atomic_file(path, writer)
