"""Write generated reports and selected weights without partial replacements."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile

import torch


def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    atomic_write(path, lambda p: p.write_text(text, encoding="utf-8"))


def write_rows(path, records):
    if not records:
        raise ValueError("Cannot export empty records")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(dict.fromkeys(k for row in records for k in row)))
    writer.writeheader()
    writer.writerows(records)
    atomic_write(path, lambda p: p.write_text(buffer.getvalue(), encoding="utf-8-sig"))


def save_checkpoint(path, payload):
    atomic_write(path, lambda p: torch.save(payload, p))
