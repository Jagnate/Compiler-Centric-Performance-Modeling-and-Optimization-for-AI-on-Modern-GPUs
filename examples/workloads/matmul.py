"""Immutable Matmul semantics for the generic TileSight evaluator."""

from __future__ import annotations


WORKLOAD_NAME = "matmul"


def reference_program(case, task):
    """Return C = A @ transpose(B) for row-major B[N, K]."""

    del case, task

    def reference(a, b):
        return a @ b.T

    return reference


def output_indices(case, task):
    return list(case.get("output_indices", task["workload"].get("output_indices", [2])))


def validate_case(case, task):
    del task
    arguments = dict(case.get("factory_arguments") or {})
    for name in ("m", "n", "k"):
        value = arguments.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError("matmul case %s must define positive integer %s" % (case["case_id"], name))

