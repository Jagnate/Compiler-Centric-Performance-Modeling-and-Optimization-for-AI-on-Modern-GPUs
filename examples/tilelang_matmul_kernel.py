"""TileLang matmul source used as the initial optimization candidate."""

# TileLang evaluates T.Buffer annotations eagerly, so postponed annotations are
# intentionally not enabled in this source file.
from tilelang import language as T


def make_matmul_program(
    m: int = 2048,
    n: int = 2048,
    k: int = 2048,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
    num_stages: int = 3,
    threads: int = 128,
):
    """Construct a valid but deliberately resource-heavy FP16 matmul seed."""

    @T.prim_func
    def matmul(
        a: T.Buffer((m, k), "float16"),
        b: T.Buffer((n, k), "float16"),
        c: T.Buffer((m, n), "float16"),
    ):
        with T.Kernel(
            T.ceildiv(n, block_n),
            T.ceildiv(m, block_m),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared((block_m, block_k), "float16")
            b_shared = T.alloc_shared((block_n, block_k), "float16")
            c_local = T.alloc_fragment((block_m, block_n), "float")
            c_shared = T.alloc_shared((block_m, block_n), "float16")

            T.clear(c_local)
            for ko in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                T.copy(a[by * block_m, ko * block_k], a_shared)
                T.copy(b[bx * block_n, ko * block_k], b_shared)
                T.gemm(a_shared, b_shared, c_local, transpose_B=True)

            T.copy(c_local, c_shared)
            T.copy(c_shared, c[by * block_m, bx * block_n])

    return matmul
