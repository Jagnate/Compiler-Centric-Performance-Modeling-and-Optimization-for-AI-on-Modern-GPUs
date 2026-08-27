"""Naive fused residual-add plus weighted RMSNorm optimization seed."""

# TileLang evaluates T.Buffer annotations while @T.prim_func constructs TIR.
# Postponed annotations must therefore remain disabled in candidate sources.
from tilelang import language as T


def make_fused_add_rms_norm_program(
    rows: int = 8192,
    hidden_size: int = 4096,
    epsilon: float = 1e-6,
    block_rows: int = 1,
    block_hidden: int = 128,
    threads: int = 64,
):
    """Construct an intentionally under-tuned fused Add + RMSNorm kernel."""

    if rows <= 0 or hidden_size <= 0:
        raise ValueError("rows and hidden_size must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if block_rows <= 0 or block_hidden <= 0 or threads <= 0:
        raise ValueError("schedule parameters must be positive")
    if rows % block_rows:
        raise ValueError("rows must be divisible by block_rows")
    if hidden_size % block_hidden:
        raise ValueError("hidden_size must be divisible by block_hidden")
    if threads % 32 or threads > 1024:
        raise ValueError("threads must be a warp multiple no greater than 1024")

    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def fused_add_rms_norm(
        x: T.Buffer((rows, hidden_size), dtype),
        residual: T.Buffer((rows, hidden_size), dtype),
        weight: T.Buffer((hidden_size,), dtype),
        normalized: T.Buffer((rows, hidden_size), dtype),
        residual_out: T.Buffer((rows, hidden_size), dtype),
    ):
        with T.Kernel(T.ceildiv(rows, block_rows), threads=threads) as bx:
            x_shared = T.alloc_shared((block_rows, block_hidden), dtype)
            residual_shared = T.alloc_shared(
                (block_rows, block_hidden), dtype
            )
            weight_shared = T.alloc_shared((block_hidden,), dtype)
            square_acc = T.alloc_fragment(
                (block_rows, block_hidden), accum_dtype
            )
            row_scale = T.alloc_fragment((block_rows,), accum_dtype)

            T.clear(square_acc)
            for ko in T.Serial(hidden_size // block_hidden):
                T.copy(
                    x[bx * block_rows, ko * block_hidden],
                    x_shared,
                )
                T.copy(
                    residual[bx * block_rows, ko * block_hidden],
                    residual_shared,
                )
                for i, j in T.Parallel(block_rows, block_hidden):
                    residual_shared[i, j] = (
                        x_shared[i, j] + residual_shared[i, j]
                    )
                    residual_value = T.Cast(
                        accum_dtype, residual_shared[i, j]
                    )
                    square_acc[i, j] += residual_value * residual_value
                T.copy(
                    residual_shared,
                    residual_out[bx * block_rows, ko * block_hidden],
                )

            T.reduce_sum(square_acc, row_scale, dim=1)
            for i in T.Parallel(block_rows):
                row_scale[i] = T.rsqrt(
                    row_scale[i] / hidden_size + epsilon
                )

            # This second pass deliberately reloads and recomputes the residual
            # sum. It is valid, but leaves clear fusion and retention headroom.
            for ko in T.Serial(hidden_size // block_hidden):
                T.copy(
                    x[bx * block_rows, ko * block_hidden],
                    x_shared,
                )
                T.copy(
                    residual[bx * block_rows, ko * block_hidden],
                    residual_shared,
                )
                T.copy(weight[ko * block_hidden], weight_shared)
                for i, j in T.Parallel(block_rows, block_hidden):
                    residual_shared[i, j] = (
                        x_shared[i, j] + residual_shared[i, j]
                    )
                    residual_value = T.Cast(
                        accum_dtype, residual_shared[i, j]
                    )
                    residual_shared[i, j] = (
                        residual_value
                        * row_scale[i]
                        * T.Cast(accum_dtype, weight_shared[j])
                    )
                T.copy(
                    residual_shared,
                    normalized[bx * block_rows, ko * block_hidden],
                )

    return fused_add_rms_norm
