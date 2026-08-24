"""Immutable weighted RMSNorm semantics for generic evaluation."""

from __future__ import annotations


WORKLOAD_NAME = "weighted-rms-norm"


def reference_program(case, task):
    """Return Y = X * rsqrt(mean(X^2) + epsilon) * weight."""

    del task
    epsilon = float(case.get("factory_arguments", {}).get("epsilon", 1e-6))

    def reference(x, weight):
        import torch

        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + epsilon)
        return (normalized * weight.float()).to(dtype=x.dtype)

    return reference


def output_indices(case, task):
    return list(
        case.get("output_indices", task["workload"].get("output_indices", [2]))
    )


def validate_case(case, task):
    del task
    arguments = dict(case.get("factory_arguments") or {})
    for name in ("rows", "hidden_size"):
        value = arguments.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                "RMSNorm case %s must define positive integer %s"
                % (case["case_id"], name)
            )
    epsilon = arguments.get("epsilon")
    if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool):
        raise ValueError(
            "RMSNorm case %s must define numeric epsilon" % case["case_id"]
        )
    if float(epsilon) <= 0:
        raise ValueError("RMSNorm epsilon must be positive")
