"""TileLang NHWC Conv2D source used as an optimization seed."""

# TileLang evaluates T.Buffer annotations while @T.prim_func constructs TIR.
# Postponed annotations must therefore remain disabled in candidate sources.
from tilelang import language as T


def make_conv2d_program(
    batch: int = 32,
    in_height: int = 56,
    in_width: int = 56,
    in_channels: int = 64,
    out_channels: int = 128,
    kernel_size: int = 3,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 1,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 32,
    num_stages: int = 1,
    threads: int = 128,
):
    """Construct a conservative implicit-GEMM FP16 Conv2D baseline."""

    semantic_values = (
        batch,
        in_height,
        in_width,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
    )
    if any(value <= 0 for value in semantic_values):
        raise ValueError("Conv2D dimensions, stride, and dilation must be positive")
    if padding < 0:
        raise ValueError("Conv2D padding cannot be negative")
    if min(block_m, block_n, block_k, num_stages, threads) <= 0:
        raise ValueError("schedule parameters must be positive")

    out_height = (
        in_height + 2 * padding - dilation * (kernel_size - 1) - 1
    ) // stride + 1
    out_width = (
        in_width + 2 * padding - dilation * (kernel_size - 1) - 1
    ) // stride + 1
    if out_height <= 0 or out_width <= 0:
        raise ValueError("Conv2D configuration produces an empty output")

    reduction_size = kernel_size * kernel_size * in_channels
    output_rows = batch * out_height * out_width
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def conv2d(
        data: T.Buffer(
            (batch, in_height, in_width, in_channels), dtype
        ),
        kernel: T.Buffer(
            (kernel_size, kernel_size, in_channels, out_channels), dtype
        ),
        output: T.Buffer(
            (batch, out_height, out_width, out_channels), dtype
        ),
    ):
        with T.Kernel(
            T.ceildiv(out_channels, block_n),
            T.ceildiv(output_rows, block_m),
            threads=threads,
        ) as (bx, by):
            data_shared = T.alloc_shared((block_m, block_k), dtype)
            kernel_shared = T.alloc_shared((block_k, block_n), dtype)
            output_local = T.alloc_fragment((block_m, block_n), accum_dtype)
            output_shared = T.alloc_shared((block_m, block_n), dtype)

            kernel_flat = T.Tensor(
                (reduction_size, out_channels), dtype, kernel.data
            )
            output_flat = T.Tensor(
                (output_rows, out_channels), dtype, output.data
            )

            T.clear(output_local)
            for ko in T.Pipelined(
                T.ceildiv(reduction_size, block_k),
                num_stages=num_stages,
            ):
                T.im2col(
                    data,
                    data_shared,
                    by,
                    ko,
                    kernel_size,
                    stride,
                    dilation,
                    padding,
                )
                T.copy(
                    kernel_flat[ko * block_k, bx * block_n],
                    kernel_shared,
                )
                T.gemm(data_shared, kernel_shared, output_local)

            T.copy(output_local, output_shared)
            T.copy(
                output_shared,
                output_flat[by * block_m, bx * block_n],
            )

    return conv2d
