#!/usr/bin/env python3
"""Test-only implementation of the source evaluator subprocess contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("model", "measure", "profile"), required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    source_path = Path(request["source"]["path"])
    source_code = source_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    if digest != request["source"]["sha256"]:
        raise RuntimeError("materialized source digest does not match the request")
    if "source_code" in request["candidate"]:
        raise RuntimeError("candidate metadata must not duplicate source_code")

    if args.stage == "model":
        response = {
            "valid": True,
            "predicted_latency_ms": 1.25,
            "bottleneck": "tensor-core",
            "confidence": "high",
            "metrics": {
                "source_path_exists": source_path.is_file(),
                "secret_visible": bool(os.environ.get("TEST_API_KEY")),
                "safe_environment": os.environ.get("SAFE_TEST_ENV"),
            },
            "diagnostics": [],
        }
    elif args.stage == "measure":
        response = {
            "correct": True,
            "latency_ms": 1.0,
            "samples_ms": [0.99, 1.0, 1.01],
            "metrics": {"measurement_source": "test-contract"},
            "error": None,
        }
    else:
        response = {
            "bottleneck": "tensor-core",
            "valid": True,
            "metrics": {"profile_source": "test-contract"},
            "report_path": None,
            "error": None,
        }
    args.response.write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
