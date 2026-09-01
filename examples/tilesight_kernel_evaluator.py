#!/usr/bin/env python3
"""Generic TileSight/TileLang evaluator driven by a workload plugin."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_WORKER_RESPONSE_PREFIX = "KERNEL_OPT_WORKER_RESPONSE "

# Keep milestone profiles focused on evidence used by the controller. Nsight
# Compute's full set collects thousands of counters and can replay the kernel
# many times; these counters cover latency, memory hierarchy, occupancy,
# resource footprint, compute pipes, and shared-memory conflict diagnosis.
TILESIGHT_TARGETED_NCU_METRICS = (
    "gpu__time_duration.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_write.sum.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "lts__t_sector_hit_rate.pct",
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
    "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
    "lts__t_sectors_srcunit_tex_op_write.sum",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "launch__registers_per_thread",
    "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persistent-worker", action="store_true")
    parser.add_argument(
        "--stage", choices=("tir", "model", "measure", "profile", "final")
    )
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path)
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()
    if args.persistent_worker:
        if args.run_once or args.stage or args.request or args.response:
            parser.error("--persistent-worker cannot be combined with stage arguments")
        return args
    if args.run_once == bool(args.stage):
        parser.error("select exactly one of --run-once or --stage")
    if args.request is None:
        parser.error("--request is required")
    if args.stage and args.response is None:
        parser.error("--response is required with --stage")
    return args


def load_request(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("request must be a JSON object")
    return value


def runtime_config(request: Mapping[str, Any]) -> Dict[str, Any]:
    evaluator = request["task"].get("evaluator", {})
    runtime = evaluator.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("task.evaluator.runtime must be a JSON object")
    return dict(runtime)


def resolve_cases(
    request: Mapping[str, Any], stage: str
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return correctness cases and the primary performance objective case."""

    task = request["task"]
    workload = dict(task.get("workload") or {})
    configuration = runtime_config(request)
    raw_search = configuration.get("search_cases")
    if raw_search is None:
        raw_search = [{"case_id": "primary"}]
    search_cases = _merge_cases(workload, raw_search, "search")
    if not search_cases:
        raise ValueError("the evaluator needs at least one search case")
    primary = search_cases[0]
    if stage != "final":
        return search_cases, primary

    final_cases = _merge_cases(
        workload, configuration.get("final_cases") or [], "final"
    )
    combined = []
    seen = set()
    for case in search_cases + final_cases:
        identifier = str(case["case_id"])
        if identifier not in seen:
            combined.append(case)
            seen.add(identifier)
    return combined, primary


def load_workload_plugin(request: Mapping[str, Any]):
    configuration = runtime_config(request)
    path_value = configuration.get("plugin")
    if not path_value:
        raise ValueError("task.evaluator.runtime.plugin is required")
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError("workload plugin does not exist: %s" % path)
    module = _load_module(path, "kernel_workload_" + _path_digest(path))
    reference = getattr(module, "reference_program", None)
    if not callable(reference):
        raise ValueError("workload plugin must define reference_program(case, task)")
    return module


def load_candidate_module(request: Mapping[str, Any]):
    source_path = Path(request["source"]["path"]).resolve()
    source_code = source_path.read_text(encoding="utf-8")
    digest = hashlib.sha256(source_code.encode("utf-8")).hexdigest()
    if digest != request["source"]["sha256"]:
        raise ValueError("candidate source digest does not match the request")
    name = "kernel_candidate_" + request["candidate"]["candidate_id"]
    return _load_module(source_path, name)


def build_program(
    request: Mapping[str, Any],
    candidate_module,
    plugin,
    case: Mapping[str, Any],
):
    task = request["task"]
    factory = getattr(candidate_module, task["entrypoint"], None)
    if not callable(factory):
        raise ValueError(
            "candidate does not define callable entrypoint %r" % task["entrypoint"]
        )
    validator = getattr(plugin, "validate_case", None)
    if callable(validator):
        validator(dict(case), task)
    custom_builder = getattr(plugin, "build_program", None)
    if callable(custom_builder):
        return custom_builder(factory, dict(case), task)
    arguments = dict(case.get("factory_arguments") or {})
    return factory(**arguments)


def output_indices(plugin, case: Mapping[str, Any], task: Mapping[str, Any]) -> List[int]:
    resolver = getattr(plugin, "output_indices", None)
    if callable(resolver):
        values = resolver(dict(case), task)
    else:
        values = case.get("output_indices", task.get("workload", {}).get("output_indices"))
    if not isinstance(values, list) or not values:
        raise ValueError("each case needs a non-empty output_indices list")
    return [int(item) for item in values]


def resolve_environment(request: Mapping[str, Any]):
    from tilesight.tir_interface.utils import create_device_arch, get_device_name

    task = request["task"]
    target_config = dict(task.get("target") or {})
    configuration = runtime_config(request)
    device_name = get_device_name()
    arch = create_device_arch(device_name)
    profile = str(configuration.get("architecture_profile", "ncu"))
    if profile == "ncu":
        arch.set_to_ncu()
    elif profile == "spec":
        arch.set_to_spec()
    elif profile == "microbench":
        arch.set_to_microbench()
    else:
        raise ValueError(
            "architecture_profile must be ncu, spec, or microbench"
        )
    target = target_config.get("tilelang_target", "auto")
    if bool(configuration.get("require_target_match", True)):
        _validate_target_match(target_config, arch, target)
    metadata = {
        "device_name": device_name,
        "tilesight_architecture": str(getattr(arch, "core", "unknown")),
        "architecture_profile": profile,
        "tilelang_target": target,
    }
    return arch, target, metadata


def model_candidate(
    request: Mapping[str, Any],
    *,
    collect_ptxas: bool,
    modeler: Optional[Callable[..., Any]] = None,
    environment_resolver: Callable[[Mapping[str, Any]], Any] = resolve_environment,
):
    if modeler is None:
        from tilesight.tir_interface.tilelang_integration import model_tilelang_program

        modeler = model_tilelang_program
    plugin = load_workload_plugin(request)
    candidate_module = load_candidate_module(request)
    _cases, primary = resolve_cases(request, "model")
    program = build_program(request, candidate_module, plugin, primary)
    arch, target, environment = environment_resolver(request)
    target_config = request["task"].get("target", {})
    configuration = runtime_config(request)
    result, lowered, snapshots = modeler(
        program,
        arch,
        l2_hit_rate=target_config.get("l2_hit_rate"),
        ptxas_registers=target_config.get("ptxas_registers"),
        target=target,
        collect_ptxas=collect_ptxas,
        cache_random_seed=int(configuration.get("cache_random_seed", 0)),
        write_policy=target_config.get("write_policy", "write-through"),
    )
    return program, result, lowered, snapshots, target, environment, primary


def tir_response(
    request: Mapping[str, Any],
    *,
    extractor: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Extract compact source-level TIR facts without running TileSight.

    The source PrimFunc is sufficient for generation guidance and avoids a
    second lowering/compilation pass for every measured candidate.  CUDA Event
    remains authoritative for correctness and latency.
    """

    try:
        if extractor is None:
            from tilesight.tir_interface import extract_tir

            extractor = extract_tir
        plugin = load_workload_plugin(request)
        candidate_module = load_candidate_module(request)
        _cases, primary = resolve_cases(request, "measure")
        program = build_program(
            request, candidate_module, plugin, primary
        )
        extracted = extractor(program)
        features, diagnostics = _compact_tir_features(extracted)
        features.update(
            {
                "analysis_source": "source-level-tir",
                "primary_case_id": primary["case_id"],
                "target": dict(request["task"].get("target") or {}),
            }
        )
        return {
            "valid": True,
            "features": _json_safe(features),
            "diagnostics": diagnostics,
            "error": None,
        }
    except Exception as error:
        return {
            "valid": False,
            "features": {},
            "diagnostics": [
                "%s: %s" % (type(error).__name__, error)
            ],
            "error": "%s: %s" % (type(error).__name__, error),
        }


def _compact_tir_features(program) -> Tuple[Dict[str, Any], List[str]]:
    operations = list(program.walk_operations())
    loops = list(program.root.walk_loops())
    resource_names = (
        "global_read_bytes",
        "global_write_bytes",
        "l2_read_bytes",
        "l2_write_bytes",
        "smem_read_bytes",
        "smem_write_bytes",
        "tensor_flops",
        "cuda_flops",
        "sfu_ops",
        "integer_ops",
        "reduction_ops",
        "sync_ops",
    )
    totals = {
        name: float(
            sum(float(getattr(item.resources, name, 0.0)) for item in operations)
        )
        for name in resource_names
    }
    kind_counts: Dict[str, int] = {}
    for operation in operations:
        kind = str(operation.kind)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
    buffers = [
        {
            "name": str(buffer.name),
            "scope": str(buffer.scope),
            "shape": [str(item) for item in buffer.shape],
            "dtype": str(buffer.dtype),
            "size_bytes": buffer.size_bytes,
            "is_parameter": bool(buffer.is_parameter),
        }
        for buffer in list(program.buffers.values())[:24]
    ]
    loop_features = [
        {
            "name": str(loop.name),
            "extent": int(loop.extent),
            "pipeline_depth": int(loop.pipeline_depth),
            "schedule_policy": str(loop.schedule_policy),
            "loop_carried_dependency_count": len(
                loop.loop_carried_dependencies
            ),
        }
        for loop in loops[:24]
    ]
    operation_features = [
        {
            "name": str(operation.name),
            "kind": str(operation.kind),
            "pipeline_stage": int(operation.pipeline_stage),
            "pipeline_order": operation.pipeline_order,
            "is_async": bool(operation.is_async),
            "read_scopes": sorted(
                {str(access.scope) for access in operation.reads}
            ),
            "write_scopes": sorted(
                {str(access.scope) for access in operation.writes}
            ),
            "dependency_count": len(operation.dependencies),
            "loop_carried_dependency_count": len(
                operation.loop_carried_dependencies
            ),
        }
        for operation in operations[:32]
    ]
    structural_payload = {
        "grid_shape": list(program.grid_shape),
        "threads_per_block": int(program.threads_per_block),
        "buffers": buffers,
        "loops": loop_features,
        "operations": operation_features,
        "operation_kind_counts": kind_counts,
    }
    fingerprint = hashlib.sha256(
        json.dumps(structural_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    diagnostics = [
        "%s %s: %s" % (item.level, item.code, item.message)
        for item in list(program.diagnostics)[:16]
    ]
    features = {
        "symbol": str(program.symbol),
        "grid_shape": list(program.grid_shape),
        "threads_per_block": int(program.threads_per_block),
        "warps_per_block": int(program.warps_per_block),
        "estimated_shared_memory_bytes": float(program.smem_footprint),
        "estimated_registers_per_thread": float(program.reg_footprint),
        "buffers": buffers,
        "loops": loop_features,
        "operations": operation_features,
        "operation_kind_counts": kind_counts,
        "resource_totals_per_source_iteration": totals,
        "structural_fingerprint": fingerprint,
        "diagnostic_count": len(program.diagnostics),
    }
    return features, diagnostics


def model_response(
    request: Mapping[str, Any],
    *,
    modeler: Optional[Callable[..., Any]] = None,
    environment_resolver: Callable[[Mapping[str, Any]], Any] = resolve_environment,
) -> Dict[str, Any]:
    collect_ptxas = bool(runtime_config(request).get("model_collect_ptxas", False))
    try:
        (
            _program,
            result,
            lowered,
            snapshots,
            target,
            environment,
            primary,
        ) = model_candidate(
            request,
            collect_ptxas=collect_ptxas,
            modeler=modeler,
            environment_resolver=environment_resolver,
        )
    except Exception as error:
        return _model_failure(error)

    metrics = result.metrics
    program = result.program
    model_input = result.model_input
    diagnostics = [
        "%s %s: %s" % (item.level, item.code, item.message)
        for item in program.diagnostics
    ]
    kernel_source = str(getattr(lowered, "kernel_source", "") or "")
    compiled_source_sha256 = (
        hashlib.sha256(kernel_source.encode("utf-8")).hexdigest()
        if kernel_source
        else None
    )
    compiled_identity = {
        "compiled_source_sha256": compiled_source_sha256,
        "target": target,
        "grid_shape": [int(item) for item in model_input.grids],
        "threads_per_block": int(program.threads_per_block),
        "shared_memory_per_block": float(metrics.smem_footprint),
    }
    ptxas = program.metadata.get("model_enrichment", {}).get("ptxas", {})
    resource_provenance = dict(getattr(model_input, "provenance", {}) or {})
    register_source = str(resource_provenance.get("registers", "unknown"))
    response_metrics = {
        "ddr_util": float(metrics.ddr_util),
        "l2_hit_rate": float(metrics.l2_hit_rate),
        "l2_util": float(metrics.l2_util),
        "smem_util": float(metrics.smem_util),
        "tensor_util": float(metrics.tensor_util),
        "cuda_util": float(metrics.cuda_util),
        "sfu_util": float(metrics.sfu_util),
        "shared_memory_per_block": float(metrics.smem_footprint),
        "registers_per_thread": float(metrics.reg_footprint),
        "grid_shape": [int(item) for item in model_input.grids],
        "threads_per_block": int(program.threads_per_block),
        "tiles_per_sm": int(model_input.tiles_per_sm),
        "logical_global_write_bytes_per_cta": float(
            getattr(model_input, "logical_global_write_io", 0.0)
        ),
        "l2_write_bytes_per_cta": float(getattr(model_input, "l2_write_io", 0.0)),
        "dram_write_bytes_per_cta": float(getattr(model_input, "ddr_write_io", 0.0)),
        "spill_load_bytes_per_cta": float(getattr(model_input, "spill_load_io", 0.0)),
        "spill_store_bytes_per_cta": float(getattr(model_input, "spill_store_io", 0.0)),
        "model_source": "tilesight-tir-interface",
        "target": target,
        "environment": environment,
        "primary_case_id": primary["case_id"],
        "compiled_source_sha256": compiled_source_sha256,
        "compiled_identity_sha256": (
            hashlib.sha256(
                json.dumps(compiled_identity, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if compiled_source_sha256
            else None
        ),
        "compiled_identity": compiled_identity,
        "resource_provenance": resource_provenance,
        "ptxas": ptxas,
        "model_collect_ptxas": collect_ptxas,
        "register_source": register_source,
        "resource_fidelity": (
            "compiled" if register_source in {"compiled-ptxas", "external-ptxas"}
            else "fast-screening"
        ),
        "captured_passes": {
            "before": sorted(getattr(snapshots, "before", {})),
            "after": sorted(getattr(snapshots, "after", {})),
        },
    }
    return {
        "valid": True,
        "predicted_latency_ms": float(metrics.latency) * 1000.0,
        "bottleneck": infer_model_bottleneck(metrics),
        "confidence": (
            "high"
            if not diagnostics
            and register_source in {"compiled-ptxas", "external-ptxas"}
            else "medium"
        ),
        "metrics": _json_safe(response_metrics),
        "diagnostics": diagnostics,
    }


def measure_response(
    request: Mapping[str, Any],
    *,
    final: bool = False,
    validation_runner: Optional[Callable[..., Any]] = None,
    environment_resolver: Callable[[Mapping[str, Any]], Any] = resolve_environment,
) -> Dict[str, Any]:
    if validation_runner is None:
        from tilesight.tir_interface.validation import run_tilelang_validation

        validation_runner = run_tilelang_validation
    stage = "final" if final else "measure"
    fresh_process = not bool(request.get("_persistent_worker", False))
    try:
        plugin = load_workload_plugin(request)
        candidate_module = load_candidate_module(request)
        cases, primary = resolve_cases(request, stage)
        _arch, target, environment = environment_resolver(request)
        configuration = runtime_config(request)
        repeats_name = "final_repeats" if final else "measurement_repeats"
        repeats = int(configuration.get(repeats_name, 1))
        if repeats <= 0:
            raise ValueError("%s must be positive" % repeats_name)

        case_results = []
        samples: List[float] = []
        supports_benchmark_repeats = _accepts_keyword_argument(
            validation_runner, "benchmark_repeats"
        )
        for case in cases:
            identifier = str(case["case_id"])
            program = build_program(request, candidate_module, plugin, case)
            reference = plugin.reference_program(dict(case), request["task"])
            validation_options = _validation_options(case, configuration)
            if (
                identifier == str(primary["case_id"])
                and supports_benchmark_repeats
            ):
                validation_options["benchmark_repeats"] = repeats
            first = validation_runner(
                program,
                target,
                output_indices=output_indices(plugin, case, request["task"]),
                reference_program=reference,
                check_correctness=True,
                benchmark=identifier == str(primary["case_id"]),
                **validation_options,
            )
            if first.correctness != "passed":
                raise RuntimeError(
                    "case %s did not report passed correctness" % identifier
                )
            if identifier == str(primary["case_id"]):
                if first.benchmark_latency_ms is None:
                    raise RuntimeError("primary case did not return benchmark latency")
                batched_samples = list(
                    getattr(first, "benchmark_samples_ms", None) or []
                )
                if batched_samples:
                    if len(batched_samples) != repeats:
                        raise RuntimeError(
                            "primary benchmark returned %d samples; expected %d"
                            % (len(batched_samples), repeats)
                        )
                    samples.extend(float(item) for item in batched_samples)
                else:
                    samples.append(float(first.benchmark_latency_ms))
                remaining_repeats = 0 if batched_samples else repeats - 1
                for _ in range(remaining_repeats):
                    repeat_program = build_program(
                        request, candidate_module, plugin, case
                    )
                    repeated = validation_runner(
                        repeat_program,
                        target,
                        output_indices=output_indices(
                            plugin, case, request["task"]
                        ),
                        check_correctness=False,
                        benchmark=True,
                        **_validation_options(case, configuration),
                    )
                    if repeated.benchmark_latency_ms is None:
                        raise RuntimeError(
                            "repeat for primary case did not return benchmark latency"
                        )
                    samples.append(float(repeated.benchmark_latency_ms))
            case_results.append(
                {
                    "case_id": identifier,
                    "factory_arguments": dict(case.get("factory_arguments") or {}),
                    "held_out": bool(case.get("held_out", False)),
                    "correctness": first.correctness,
                    "runtime": _runtime_to_dict(first),
                }
            )

        statistics_value = robust_statistics(samples)
        return {
            "correct": True,
            "latency_ms": statistics_value["median_ms"],
            "samples_ms": samples,
            "metrics": {
                "measurement_source": "cuda-events",
                "stage": stage,
                "fresh_process": fresh_process,
                "primary_case_id": primary["case_id"],
                "case_count": len(cases),
                "held_out_case_count": sum(
                    bool(case.get("held_out", False)) for case in cases
                ),
                "cases": case_results,
                "statistics": statistics_value,
                "environment": environment,
            },
            "error": None,
        }
    except Exception as error:
        return {
            "correct": False,
            "latency_ms": None,
            "samples_ms": [],
            "metrics": {"stage": stage, "fresh_process": fresh_process},
            "error": "%s: %s" % (type(error).__name__, error),
        }


def profile_response(request: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        (
            _program,
            model,
            _lowered,
            _snapshots,
            _target,
            _environment,
            _primary,
        ) = model_candidate(request, collect_ptxas=False)
        report_path, csv_path = run_ncu(request)
        from tilesight.tir_interface.validation import load_ncu_metrics

        ncu = load_ncu_metrics(
            csv_path,
            kernel_name=infer_ncu_kernel_name(model),
        )
        metrics = ncu.to_dict()
        metrics["ncu_csv_path"] = str(csv_path.resolve())
        return {
            "bottleneck": infer_ncu_bottleneck(metrics),
            "valid": True,
            "metrics": _json_safe(metrics),
            "report_path": str(report_path.resolve()),
            "error": None,
        }
    except Exception as error:
        return {
            "bottleneck": "profile-unavailable",
            "valid": False,
            "metrics": {"error_type": type(error).__name__},
            "report_path": None,
            "error": "%s: %s" % (type(error).__name__, error),
        }


def run_once(request: Mapping[str, Any]) -> None:
    from tilesight.tir_interface.validation import run_tilelang_validation

    plugin = load_workload_plugin(request)
    candidate_module = load_candidate_module(request)
    _cases, primary = resolve_cases(request, "measure")
    program = build_program(request, candidate_module, plugin, primary)
    _arch, target, _environment = resolve_environment(request)
    run_tilelang_validation(
        program,
        target,
        output_indices=output_indices(plugin, primary, request["task"]),
        run_once=True,
    )


def run_ncu(request: Mapping[str, Any]) -> Tuple[Path, Path]:
    task = request["task"]
    metadata = task.get("metadata", {})
    configuration = runtime_config(request)
    ncu = configuration.get("ncu_path") or metadata.get("ncu_path") or shutil.which("ncu")
    if not ncu:
        raise RuntimeError("ncu is not available on PATH")
    directory = Path(
        configuration.get("profile_directory")
        or metadata.get("profile_directory", "results/ncu_profiles")
    )
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    directory.mkdir(parents=True, exist_ok=True)
    candidate_id = request["candidate"]["candidate_id"]
    safe_task = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task["task_id"]))
    report_stem = (directory / (safe_task + "_" + candidate_id)).resolve()
    report_path = Path(str(report_stem) + ".ncu-rep")
    csv_path = Path(str(report_stem) + ".csv")
    command = [
        str(ncu),
        "-f",
        "-o",
        str(report_stem),
    ]
    command.extend(_ncu_collection_arguments(configuration, metadata))
    command.extend(
        [
        "--cache-control",
        str(configuration.get("ncu_cache_control", "all")),
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-once",
        "--request",
        str(Path(request["_request_path"]).resolve()),
        ]
    )
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
        [str(ncu), "--import", str(report_path), "--csv", "--page", "raw"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if exported.returncode != 0:
        raise RuntimeError("NCU export failed: %s" % exported.stdout[-4000:])
    csv_path.write_text(exported.stdout, encoding="utf-8")
    return report_path, csv_path


def _ncu_collection_arguments(
    configuration: Mapping[str, Any], metadata: Mapping[str, Any]
) -> List[str]:
    requested = configuration.get("ncu_metrics")
    if requested is None:
        requested = metadata.get("ncu_metrics")
    if requested is not None:
        if isinstance(requested, str):
            metrics = [item.strip() for item in requested.split(",") if item.strip()]
        elif isinstance(requested, Sequence):
            metrics = [str(item).strip() for item in requested if str(item).strip()]
        else:
            raise ValueError("ncu_metrics must be a list or comma-separated string")
        if not metrics:
            raise ValueError("ncu_metrics cannot be empty")
        return ["--metrics", ",".join(dict.fromkeys(metrics))]

    requested_set = str(
        configuration.get("ncu_set")
        or metadata.get("ncu_set", "tilesight-targeted")
    )
    if requested_set == "tilesight-targeted":
        return ["--metrics", ",".join(TILESIGHT_TARGETED_NCU_METRICS)]
    return ["--set", requested_set]


def robust_statistics(samples: Sequence[float]) -> Dict[str, Any]:
    values = [float(item) for item in samples]
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise ValueError("timing samples must be finite positive values")
    median = statistics.median(values)
    deviations = [abs(item - median) for item in values]
    mean = statistics.mean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "sample_count": len(values),
        "median_ms": median,
        "mean_ms": mean,
        "min_ms": min(values),
        "max_ms": max(values),
        "standard_deviation_ms": standard_deviation,
        "median_absolute_deviation_ms": statistics.median(deviations),
        "coefficient_of_variation": (
            standard_deviation / mean if mean > 0 else None
        ),
    }


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


def infer_ncu_bottleneck(metrics: Mapping[str, Any]) -> str:
    utilizations = {
        "dram-bandwidth": metrics.get("ddr_util"),
        "l2-bandwidth": metrics.get("l2_util"),
        "shared-memory": metrics.get("smem_util"),
        "tensor-core": metrics.get("tensor_util"),
        "cuda-core": metrics.get("cuda_util"),
        "sfu": metrics.get("sfu_util"),
    }
    valid = {
        name: float(value)
        for name, value in utilizations.items()
        if value is not None
    }
    return max(valid, key=valid.get) if valid else "unknown"


def infer_ncu_kernel_name(model_result) -> str:
    ptxas = model_result.program.metadata.get("model_enrichment", {}).get("ptxas", {})
    function = ptxas.get("function") or {}
    if function.get("name"):
        return str(function["name"])
    symbol = str(model_result.program.symbol)
    return symbol if symbol.endswith("_kernel") else symbol + "_kernel"


def write_response(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def evaluate_stage(stage: str, request: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate one request using the same dispatch in one-shot and worker modes."""

    if stage == "tir":
        return tir_response(request)
    if stage == "model":
        return model_response(request)
    if stage == "measure":
        return measure_response(request, final=False)
    if stage == "final":
        return measure_response(request, final=True)
    if stage == "profile":
        return profile_response(request)
    raise ValueError("unsupported evaluator stage %r" % stage)


@contextmanager
def _capture_worker_output(path: Path):
    """Redirect Python and native stdout/stderr while preserving protocol stdout."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        previous_stdout = sys.stdout
        previous_stderr = sys.stderr
        try:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            sys.stdout = stream
            sys.stderr = stream
            yield
        finally:
            stream.flush()
            sys.stdout = previous_stdout
            sys.stderr = previous_stderr
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


def _cleanup_worker_request() -> None:
    """Release request-scoped Python and CUDA caches without reimporting runtimes."""

    gc.collect()
    torch_module = sys.modules.get("torch")
    cuda = getattr(torch_module, "cuda", None) if torch_module is not None else None
    if cuda is not None:
        try:
            if cuda.is_available():
                cuda.empty_cache()
        except Exception:
            pass


def persistent_worker() -> None:
    """Serve path-only JSON requests while keeping compiler imports warm."""

    protocol_stdout = sys.stdout
    for line in sys.stdin:
        try:
            message = json.loads(line)
        except Exception as error:
            acknowledgement = {
                "request_id": None,
                "status": "error",
                "error": {"type": type(error).__name__, "message": str(error)},
            }
            protocol_stdout.write(
                _WORKER_RESPONSE_PREFIX
                + json.dumps(acknowledgement, sort_keys=True)
                + "\n"
            )
            protocol_stdout.flush()
            continue
        if message.get("action") == "shutdown":
            protocol_stdout.flush()
            os._exit(0)

        request_id = str(message.get("request_id") or "")
        output_path = Path(str(message.get("stdout")))
        acknowledgement = {"request_id": request_id, "status": "ok"}
        try:
            with _capture_worker_output(output_path):
                request_path = Path(str(message["request"]))
                response_path = Path(str(message["response"]))
                stage = str(message["stage"])
                request = load_request(request_path)
                request["_request_path"] = str(request_path)
                request["_persistent_worker"] = True
                response = evaluate_stage(stage, request)
                write_response(response_path, response)
                _cleanup_worker_request()
        except BaseException as error:
            try:
                with _capture_worker_output(output_path):
                    traceback.print_exc()
                    _cleanup_worker_request()
            except Exception:
                pass
            acknowledgement = {
                "request_id": request_id,
                "status": "error",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        protocol_stdout.write(
            _WORKER_RESPONSE_PREFIX
            + json.dumps(acknowledgement, sort_keys=True)
            + "\n"
        )
        protocol_stdout.flush()
    protocol_stdout.flush()
    os._exit(0)


def _merge_cases(
    workload: Mapping[str, Any], raw_cases: Any, kind: str
) -> List[Dict[str, Any]]:
    if not isinstance(raw_cases, list):
        raise ValueError("%s_cases must be a list" % kind)
    cases = []
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError("each %s case must be a JSON object" % kind)
        case = dict(workload)
        base_arguments = dict(workload.get("factory_arguments") or {})
        base_arguments.update(dict(raw.get("factory_arguments") or {}))
        case.update(raw)
        case["factory_arguments"] = base_arguments
        case.setdefault("case_id", "%s-%d" % (kind, index))
        case["held_out"] = kind == "final"
        cases.append(case)
    return cases


def _validation_options(
    case: Mapping[str, Any], configuration: Mapping[str, Any]
) -> Dict[str, Any]:
    def value(name: str, default: Any) -> Any:
        return case.get(name, configuration.get(name, default))

    execution_backend = str(value("execution_backend", "auto"))
    return {
        "execution_backend": execution_backend,
        "atol": float(value("atol", 1e-2)),
        "rtol": float(value("rtol", 1e-2)),
        "max_mismatched_ratio": float(value("max_mismatched_ratio", 0.01)),
        "warmup_ms": float(value("warmup_ms", 25.0)),
        "rep_ms": float(value("rep_ms", 100.0)),
        "n_warmup": int(value("n_warmup", 0)),
        "n_repeat": int(value("n_repeat", 0)),
        "benchmark_backend": str(value("benchmark_backend", "event")),
        "return_mode": str(value("return_mode", "median")),
    }


def _runtime_to_dict(value: Any) -> Dict[str, Any]:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _json_safe(converter())
    return _json_safe(dict(vars(value)))


def _model_failure(error: Exception) -> Dict[str, Any]:
    return {
        "valid": False,
        "predicted_latency_ms": None,
        "bottleneck": "compile-or-model-failure",
        "confidence": "unknown",
        "metrics": {},
        "diagnostics": ["%s: %s" % (type(error).__name__, error)],
    }


def _load_module(path: Path, name: str):
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load Python module %s" % path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _path_digest(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:12]


def _validate_target_match(
    target_config: Mapping[str, Any], arch: Any, tilelang_target: Any
) -> None:
    expected = str(target_config.get("architecture", "auto"))
    actual = str(getattr(arch, "core", "unknown"))
    if expected.lower() not in {"", "auto", "unknown"}:
        if _normalize_architecture(expected) != _normalize_architecture(actual):
            raise ValueError(
                "task architecture %r does not match active TileSight architecture %r"
                % (expected, actual)
            )

    target_text = json.dumps(tilelang_target) if isinstance(tilelang_target, dict) else str(tilelang_target)
    match = re.search(r"sm[_= -]?(\d+)", target_text.lower())
    if match is None:
        return
    try:
        from tilelang import tvm

        device = tvm.cuda(0)
        compute_version = str(getattr(device, "compute_version", "") or "")
    except Exception:
        return
    actual_sm = re.sub(r"\D", "", compute_version)
    if actual_sm and actual_sm != match.group(1):
        raise ValueError(
            "TileLang target sm_%s does not match active CUDA compute capability %s"
            % (match.group(1), compute_version)
        )


def _normalize_architecture(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        return _json_safe(converter())
    return str(value)


def _accepts_keyword_argument(function: Callable[..., Any], name: str) -> bool:
    """Return whether a validation runner supports one optional keyword."""

    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False
    if name in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def main() -> int:
    args = parse_args()
    if args.persistent_worker:
        persistent_worker()
        return 0
    request = load_request(args.request)
    request["_request_path"] = str(args.request)
    if args.run_once:
        run_once(request)
        return 0
    response = evaluate_stage(args.stage, request)
    write_response(args.response, response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
