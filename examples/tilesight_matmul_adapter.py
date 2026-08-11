#!/usr/bin/env python3
"""TileSight evaluator adapter for the source-level matmul example."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("model", "measure", "profile"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()
    if args.run_once == bool(args.stage):
        parser.error("select exactly one of --run-once or --stage")
    if args.stage and args.response is None:
        parser.error("--response is required with --stage")
    return args


def load_request(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    return value


def load_program(request: Dict[str, Any]):
    task = request["task"]
    source_path = Path(request["source"]["path"])
    module_name = "kernel_candidate_" + request["candidate"]["candidate_id"]
    specification = importlib.util.spec_from_file_location(module_name, source_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load candidate source %s" % source_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    factory = getattr(module, task["entrypoint"])
    arguments = dict(task.get("workload", {}).get("factory_arguments") or {})
    return factory(**arguments)


def resolve_environment(request: Dict[str, Any]):
    from tilesight.tir_interface.utils import create_device_arch, get_device_name

    task = request["task"]
    target = task.get("target", {})
    device_name = get_device_name()
    arch = create_device_arch(device_name).set_to_ncu()
    tilelang_target = target.get("tilelang_target", "auto")
    return arch, tilelang_target


def model_candidate(request: Dict[str, Any], collect_ptxas: bool = False):
    from tilesight.tir_interface.tilelang_integration import model_tilelang_program

    program = load_program(request)
    arch, target = resolve_environment(request)
    write_policy = request["task"].get("target", {}).get(
        "write_policy", "write-through"
    )
    result, lowered, snapshots = model_tilelang_program(
        program,
        arch,
        target=target,
        collect_ptxas=collect_ptxas,
        write_policy=write_policy,
    )
    return program, result, lowered, snapshots, target


def model_response(request: Dict[str, Any]) -> Dict[str, Any]:
    try:
        _program, result, _lowered, _snapshots, _target = model_candidate(request)
    except Exception as error:
        return {
            "valid": False,
            "predicted_latency_ms": None,
            "bottleneck": "compile-or-model-failure",
            "confidence": "unknown",
            "metrics": {},
            "diagnostics": ["%s: %s" % (type(error).__name__, error)],
        }
    metrics = result.metrics
    bottleneck = infer_model_bottleneck(metrics)
    diagnostics = [
        "%s: %s" % (item.code, item.message)
        for item in result.program.diagnostics
    ]
    confidence = "high" if not diagnostics else "medium"
    return {
        "valid": True,
        "predicted_latency_ms": float(metrics.latency) * 1000.0,
        "bottleneck": bottleneck,
        "confidence": confidence,
        "metrics": {
            "ddr_util": float(metrics.ddr_util),
            "l2_hit_rate": float(metrics.l2_hit_rate),
            "l2_util": float(metrics.l2_util),
            "smem_util": float(metrics.smem_util),
            "tensor_util": float(metrics.tensor_util),
            "cuda_util": float(metrics.cuda_util),
            "sfu_util": float(metrics.sfu_util),
            "shared_memory_per_block": float(metrics.smem_footprint),
            "registers_per_thread": float(metrics.reg_footprint),
            "model_source": "tilesight-tir-interface"
        },
        "diagnostics": diagnostics,
    }


def measure_response(request: Dict[str, Any]) -> Dict[str, Any]:
    from tilesight.tir_interface.validation import run_tilelang_validation

    try:
        program = load_program(request)
        _arch, target = resolve_environment(request)
        workload = request["task"].get("workload", {})
        runtime = run_tilelang_validation(
            program,
            target,
            output_indices=list(workload.get("output_indices", [2])),
            reference_program=matmul_reference,
            check_correctness=True,
            benchmark=True,
            atol=float(workload.get("atol", 1e-2)),
            rtol=float(workload.get("rtol", 1e-2)),
            max_mismatched_ratio=float(
                workload.get("max_mismatched_ratio", 0.01)
            ),
            warmup_ms=float(workload.get("warmup_ms", 25.0)),
            rep_ms=float(workload.get("rep_ms", 100.0)),
            benchmark_backend=str(workload.get("benchmark_backend", "event")),
            return_mode=str(workload.get("return_mode", "median")),
        )
    except Exception as error:
        return {
            "correct": False,
            "latency_ms": None,
            "samples_ms": [],
            "metrics": {},
            "error": "%s: %s" % (type(error).__name__, error),
        }
    return {
        "correct": runtime.correctness == "passed",
        "latency_ms": runtime.benchmark_latency_ms,
        "samples_ms": [runtime.benchmark_latency_ms],
        "metrics": runtime.to_dict(),
        "error": None,
    }


def profile_response(request: Dict[str, Any]) -> Dict[str, Any]:
    try:
        _program, model, _lowered, _snapshots, _target = model_candidate(
            request, collect_ptxas=True
        )
        report_path, csv_path = run_ncu(request)
        from tilesight.tir_interface.validation import load_ncu_metrics

        ncu = load_ncu_metrics(
            csv_path,
            kernel_name=infer_ncu_kernel_name(model),
        )
        return {
            "bottleneck": infer_ncu_bottleneck(ncu.to_dict()),
            "valid": True,
            "metrics": ncu.to_dict(),
            "report_path": str(report_path.resolve()),
            "error": None,
        }
    except Exception as error:
        return {
            "bottleneck": "profile-unavailable",
            "valid": False,
            "metrics": {"error": "%s: %s" % (type(error).__name__, error)},
            "report_path": None,
            "error": "%s: %s" % (type(error).__name__, error),
        }


def run_ncu(request: Dict[str, Any]) -> Tuple[Path, Path]:
    metadata = request["task"].get("metadata", {})
    ncu = metadata.get("ncu_path") or shutil.which("ncu")
    if not ncu:
        raise RuntimeError("ncu is not available on PATH")
    directory = Path(metadata.get("profile_directory", "results/ncu_profiles"))
    directory.mkdir(parents=True, exist_ok=True)
    candidate_id = request["candidate"]["candidate_id"]
    report_stem = (directory / candidate_id).resolve()
    report_path = Path(str(report_stem) + ".ncu-rep")
    csv_path = Path(str(report_stem) + ".csv")
    command = [
        ncu,
        "-f",
        "-o",
        str(report_stem),
        "--set",
        str(metadata.get("ncu_set", "full")),
        "--cache-control",
        "all",
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-once",
        "--request",
        str(Path(request["_request_path"]).resolve()),
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("NCU collection failed: %s" % completed.stdout[-4000:])
    exported = subprocess.run(
        [ncu, "--import", str(report_path), "--csv", "--page", "raw"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if exported.returncode != 0:
        raise RuntimeError("NCU export failed: %s" % exported.stdout[-4000:])
    csv_path.write_text(exported.stdout, encoding="utf-8")
    return report_path, csv_path


def run_once(request: Dict[str, Any]) -> None:
    from tilesight.tir_interface.validation import run_tilelang_validation

    program = load_program(request)
    _arch, target = resolve_environment(request)
    workload = request["task"].get("workload", {})
    run_tilelang_validation(
        program,
        target,
        output_indices=list(workload.get("output_indices", [2])),
        run_once=True,
    )


def matmul_reference(a, b):
    """Immutable reference outside the API-editable kernel source."""

    return a @ b.T


def infer_model_bottleneck(metrics) -> str:
    utilizations = {
        "dram-bandwidth": float(metrics.ddr_util),
        "l2-bandwidth": float(metrics.l2_util),
        "shared-memory": float(metrics.smem_util),
        "tensor-core": float(metrics.tensor_util),
        "cuda-core": float(metrics.cuda_util),
        "sfu": float(metrics.sfu_util),
    }
    return max(utilizations, key=utilizations.get)


def infer_ncu_bottleneck(metrics: Dict[str, Any]) -> str:
    utilizations = {
        "dram-bandwidth": metrics.get("ddr_util"),
        "l2-bandwidth": metrics.get("l2_util"),
        "shared-memory": metrics.get("smem_util"),
        "tensor-core": metrics.get("tensor_util"),
        "cuda-core": metrics.get("cuda_util"),
        "sfu": metrics.get("sfu_util"),
    }
    valid = {name: float(value) for name, value in utilizations.items() if value is not None}
    return max(valid, key=valid.get) if valid else "unknown"


def infer_ncu_kernel_name(model_result) -> str:
    ptxas = model_result.program.metadata.get("model_enrichment", {}).get("ptxas", {})
    function = ptxas.get("function") or {}
    if function.get("name"):
        return str(function["name"])
    symbol = str(model_result.program.symbol)
    return symbol if symbol.endswith("_kernel") else symbol + "_kernel"


def write_response(path: Path, value: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    request = load_request(args.request)
    request["_request_path"] = str(args.request)
    if args.run_once:
        run_once(request)
        return 0
    if args.stage == "model":
        response = model_response(request)
    elif args.stage == "measure":
        response = measure_response(request)
    else:
        response = profile_response(request)
    write_response(args.response, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
