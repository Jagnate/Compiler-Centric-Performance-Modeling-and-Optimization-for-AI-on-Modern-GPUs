from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_single_record_csv(path: Path, record: dict[str, Any]) -> None:
    flat = {}
    for key, value in record.items():
        flat[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow(flat)
