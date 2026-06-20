# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def _gemm_ssr():
        # with T.block("root"):
        A_local = T.alloc_buffer((32,), "float16", scope="local")
        B_local = T.alloc_buffer((32,), "float16", scope="local")
        for ki in range(2):
            k = T.int32()
            thread_binding = T.int32()
            for i in range(4):
                A_shared = T.Buffer((3, 128, 32), "float16", scope="shared.dyn")
                T.ptx_ldmatrix(T.bool(False), 4, T.access_ptr(A_shared[k % 3, thread_binding // 32 % 2 * 64 + i * 16 + thread_binding % 32 % 16, ki * 16 + 8 * (thread_binding % 32 // 16)], 8, 1), T.access_ptr(A_local[i * 8], 8, 2))
            for i in range(4):
                B_shared = T.Buffer((3, 32, 128), "float16", scope="shared.dyn")
                T.ptx_ldmatrix(T.bool(True), 4, T.access_ptr(B_shared[k % 3, ki * 16 + thread_binding % 32 % 16, thread_binding // 64 % 2 * 64 + i * 16 + 8 * (thread_binding % 32 // 16)], 8, 1), T.access_ptr(B_local[i * 8], 8, 2))
            for i, j in T.grid(4, 4):
                C_local = T.handle("float32", "local.fragment")
                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, i * 8, B_local.data, j * 8, C_local, i * 4 * 8 + j * 8, T.bool(False))
                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, i * 8, B_local.data, j * 8 + 4, C_local, i * 4 * 8 + j * 8 + 4, T.bool(False))