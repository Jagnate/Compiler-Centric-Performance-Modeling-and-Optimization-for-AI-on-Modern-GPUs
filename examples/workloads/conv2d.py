"""Immutable NHWC/HWIO Conv2D semantics for generic evaluation."""

from __future__ import annotations


WORKLOAD_NAME = "conv2d-nhwc-hwio"


def reference_program(case, task):
    del task
    arguments = dict(case.get("factory_arguments") or {})
    stride = int(arguments.get("stride", 1))
    padding = int(arguments.get("padding", 0))
    dilation = int(arguments.get("dilation", 1))

    def reference(data, kernel):
        import torch.nn.functional as functional

        data_nchw = data.permute(0, 3, 1, 2)
        kernel_oihw = kernel.permute(3, 2, 0, 1)
        output = functional.conv2d(
            data_nchw,
            kernel_oihw,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        return output.permute(0, 2, 3, 1)

    return reference


def output_indices(case, task):
    return list(
        case.get("output_indices", task["workload"].get("output_indices", [2]))
    )


def validate_case(case, task):
    del task
    arguments = dict(case.get("factory_arguments") or {})
    positive = (
        "batch",
        "in_height",
        "in_width",
        "in_channels",
        "out_channels",
        "kernel_size",
        "stride",
        "dilation",
    )
    for name in positive:
        value = arguments.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(
                "Conv2D case %s must define positive integer %s"
                % (case["case_id"], name)
            )
    padding = arguments.get("padding")
    if not isinstance(padding, int) or isinstance(padding, bool) or padding < 0:
        raise ValueError(
            "Conv2D case %s must define non-negative integer padding"
            % case["case_id"]
        )
    output_height = (
        arguments["in_height"]
        + 2 * padding
        - arguments["dilation"] * (arguments["kernel_size"] - 1)
        - 1
    ) // arguments["stride"] + 1
    output_width = (
        arguments["in_width"]
        + 2 * padding
        - arguments["dilation"] * (arguments["kernel_size"] - 1)
        - 1
    ) // arguments["stride"] + 1
    if output_height <= 0 or output_width <= 0:
        raise ValueError("Conv2D case produces an empty output")
