"""Immutable BSHD Flash Attention forward semantics for generic evaluation."""

from __future__ import annotations


WORKLOAD_NAME = "flash-attention-forward-bshd"


def reference_program(case, task):
    del task
    is_causal = bool(case.get("factory_arguments", {}).get("is_causal", False))

    def reference(q, k, v):
        import torch
        import torch.nn.functional as functional

        dim = q.size(-1)
        scores = torch.einsum("bqhd,bkhd->bhqk", q, k) / (float(dim) ** 0.5)
        if is_causal:
            query_length = q.size(1)
            key_length = k.size(1)
            mask = torch.ones(
                (query_length, key_length),
                dtype=torch.bool,
                device=scores.device,
            ).tril()
            scores = scores.masked_fill(
                ~mask[None, None, :, :], float("-inf")
            )
        attention = functional.softmax(scores, dim=-1)
        return torch.einsum("bhqk,bkhd->bqhd", attention, v)

    return reference


def output_indices(case, task):
    return list(case.get("output_indices", task["workload"].get("output_indices", [3])))


def validate_case(case, task):
    del task
    arguments = dict(case.get("factory_arguments") or {})
    for name in ("batch", "heads", "seq_len", "dim"):
        value = arguments.get(name)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(
                "flash attention case %s must define positive integer %s"
                % (case["case_id"], name)
            )
    if arguments["dim"] % 16:
        raise ValueError("flash attention dim must be a multiple of 16")

