#!/usr/bin/env python3
"""Test-only implementation of the source evaluator subprocess contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


WORKER_RESPONSE_PREFIX = "KERNEL_OPT_WORKER_RESPONSE "


def evaluate(stage: str, request_path: Path, response_path: Path, request_index: int = 1) -> None:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    source_path = Path(request["source"]["path"])
    source_code = source_path.read_text(encoding="utf-8")
    if "KERNEL_OPT_TEST_WORKER_EXIT" in source_code:
        os._exit(17)
    digest = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    if digest != request["source"]["sha256"]:
        raise RuntimeError("materialized source digest does not match the request")
    if "source_code" in request["candidate"]:
        raise RuntimeError("candidate metadata must not duplicate source_code")

    worker = {"worker_pid": os.getpid(), "worker_request_index": request_index}
    if stage == "model":
        response = {
            "valid": True,
            "predicted_latency_ms": 1.25,
            "bottleneck": "tensor-core",
            "confidence": "high",
            "metrics": {
                "source_path_exists": source_path.is_file(),
                "secret_visible": bool(os.environ.get("TEST_API_KEY")),
                "safe_environment": os.environ.get("SAFE_TEST_ENV"),
                **worker,
            },
            "diagnostics": [],
        }
    elif stage in {"measure", "final"}:
        response = {
            "correct": True,
            "latency_ms": 1.0,
            "samples_ms": [0.99, 1.0, 1.01],
            "metrics": {
                "measurement_source": "test-contract",
                "stage": stage,
                **worker,
            },
            "error": None,
        }
    else:
        response = {
            "bottleneck": "tensor-core",
            "valid": True,
            "metrics": {"profile_source": "test-contract", **worker},
            "report_path": None,
            "error": None,
        }
    response_path.write_text(
        json.dumps(response, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def persistent_worker() -> None:
    request_index = 0
    for line in sys.stdin:
        message = json.loads(line)
        if message.get("action") == "shutdown":
            sys.stdout.flush()
            os._exit(0)
        request_index += 1
        request_id = str(message["request_id"])
        try:
            evaluate(
                str(message["stage"]),
                Path(message["request"]),
                Path(message["response"]),
                request_index=request_index,
            )
            acknowledgement = {"request_id": request_id, "status": "ok"}
        except Exception as error:
            acknowledgement = {
                "request_id": request_id,
                "status": "error",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        print(WORKER_RESPONSE_PREFIX + json.dumps(acknowledgement), flush=True)
    os._exit(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-worker", action="store_true")
    parser.add_argument(
        "--stage", choices=("model", "measure", "profile", "final")
    )
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args()
    if args.persistent_worker:
        persistent_worker()
        return 0
    if args.stage is None or args.request is None or args.response is None:
        parser.error("one-shot mode requires --stage, --request, and --response")
    evaluate(args.stage, args.request, args.response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
