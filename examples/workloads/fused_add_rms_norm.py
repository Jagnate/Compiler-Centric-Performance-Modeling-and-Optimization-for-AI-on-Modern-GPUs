"""Immutable fused residual-add plus weighted RMSNorm semantics."""

from __future__ import annotations


WORKLOAD_NAME = "fused-add-weighted-rms-norm"


def reference_program(case, task):
    """Return normalized(X + residual) and the FP16 residual sum."""

    del task
    epsilon = float(case.get("factory_arguments", {}).get("epsilon", 1e-6))

    def reference(x, residual, weight):
        import torch

        residual_sum = x + residual
        residual_float = residual_sum.float()
        variance = residual_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = residual_float * torch.rsqrt(variance + epsilon)
        normalized = normalized * weight.float()
        return normalized.to(dtype=x.dtype), residual_sum

    return reference


def output_indices(case, task):
    return list(
        case.get(
            "output_indices", task["workload"].get("output_indices", [3, 4])
        )
    )


def validate_case(case, task):
    del task
    arguments = dict(case.get("factory_arguments") or {})
    for name in ("rows", "hidden_size"):
        value = arguments.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                "Fused Add + RMSNorm case %s must define positive integer %s"
                % (case["case_id"], name)
            )
    epsilon = arguments.get("epsilon")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise ValueError(
            "Fused Add + RMSNorm case %s must define numeric epsilon"
            % case["case_id"]
        )
    if float(epsilon) <= 0:
        raise ValueError("Fused Add + RMSNorm epsilon must be positive")
