"""Official-structure TileLang Flash Attention optimization seed."""

# TileLang evaluates T.Buffer annotations while @T.prim_func constructs TIR.
# Postponed annotations must therefore remain disabled in candidate sources.
from tilelang import language as T


def make_flash_attention_program(
    batch: int = 1,
    heads: int = 32,
    seq_len: int = 1024,
    dim: int = 64,
    is_causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    num_stages: int = 1,
    threads: int = 128,
):
    """Construct an official-style online-softmax Flash Attention seed."""

    scale = (1.0 / dim) ** 0.5 * 1.44269504
    shape = (batch, seq_len, heads, dim)
    dtype = "float16"
    accum_dtype = "float"

    @T.prim_func
    def main(
        q: T.Buffer(shape, dtype),
        k: T.Buffer(shape, dtype),
        v: T.Buffer(shape, dtype),
        output: T.Buffer(shape, dtype),
    ):
        with T.Kernel(
            T.ceildiv(seq_len, block_m),
            heads,
            batch,
            threads=threads,
        ) as (bx, by, bz):
            q_shared = T.alloc_shared((block_m, dim), dtype)
            k_shared = T.alloc_shared((block_n, dim), dtype)
            v_shared = T.alloc_shared((block_n, dim), dtype)
            o_shared = T.alloc_shared((block_m, dim), dtype)
            acc_s = T.alloc_fragment((block_m, block_n), accum_dtype)
            acc_s_cast = T.alloc_fragment((block_m, block_n), dtype)
            acc_o = T.alloc_fragment((block_m, dim), accum_dtype)
            scores_max = T.alloc_fragment((block_m,), accum_dtype)
            scores_max_prev = T.alloc_fragment((block_m,), accum_dtype)
            scores_scale = T.alloc_fragment((block_m,), accum_dtype)
            scores_sum = T.alloc_fragment((block_m,), accum_dtype)
            logsum = T.alloc_fragment((block_m,), accum_dtype)

            # Q is invariant across all K/V tiles for this output tile.
            T.copy(
                q[bz, bx * block_m : (bx + 1) * block_m, by, :],
                q_shared,
            )
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(
                    T.ceildiv(seq_len, block_n),
                    T.ceildiv((bx + 1) * block_m, block_n),
                )
                if is_causal
                else T.ceildiv(seq_len, block_n)
            )

            for ko in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(
                    k[bz, ko * block_n : (ko + 1) * block_n, by, :],
                    k_shared,
                )
                if is_causal:
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.if_then_else(
                            bx * block_m + i >= ko * block_n + j,
                            0,
                            -T.infinity(accum_dtype),
                        )
                else:
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.if_then_else(
                            ko * block_n + j >= seq_len,
                            -T.infinity(accum_dtype),
                            0,
                        )
                T.gemm(
                    q_shared,
                    k_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_m):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_m):
                    scores_scale[i] = T.exp2(
                        scores_max_prev[i] * scale - scores_max[i] * scale
                    )
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * scale - scores_max[i] * scale
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_m):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(
                    v[bz, ko * block_n : (ko + 1) * block_n, by, :],
                    v_shared,
                )
                T.gemm(
                    acc_s_cast,
                    v_shared,
                    acc_o,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            for i, j in T.Parallel(block_m, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, o_shared)
            T.copy(
                o_shared,
                output[bz, bx * block_m : (bx + 1) * block_m, by, :],
            )

    return main
