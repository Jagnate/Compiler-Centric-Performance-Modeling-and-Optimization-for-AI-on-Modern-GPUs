# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def matmul(A: T.handle, B: T.handle, C_handle: T.handle):
        T.func_attr({"tilelang_out_idx": [-1]})
        A_1 = T.match_buffer(A, (1024, 1024), "float16", strides=(1024, 1))
        B_1 = T.match_buffer(B, (1024, 1024), "float16", strides=(1024, 1))
        C = T.match_buffer(C_handle, (1024, 1024), "float16", strides=(1024, 1))
        # with T.block("root"):
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.block("tilelang_root"):
            T.reads(A_1[by * 128, 0:993], B_1[0:993, bx * 128], C[by * 128, bx * 128])
            T.writes()
            A_shared = T.alloc_buffer((128, 32), "float16", scope="shared.dyn")
            B_shared = T.alloc_buffer((32, 128), "float16", scope="shared.dyn")
            C_local = T.alloc_buffer((128, 128), scope="local.fragment")
            T.fill(T.region(C_local[0, 0], 2, 128, 128), 0)
            for k in T.serial(32, annotations={"num_stages": 3}):
                T.copy(T.region(A_1[by * 128, k * 32], 1, 128, 32), T.region(A_shared[0, 0], 2, 128, 32))
                T.copy(T.region(B_1[k * 32, bx * 128], 1, 32, 128), T.region(B_shared[0, 0], 2, 32, 128))
                T.gemm(T.region(A_shared[0, 0], 1, 128, 32), T.region(B_shared[0, 0], 1, 32, 128), T.region(C_local[0, 0], 3, 128, 128), T.bool(False), T.bool(False), 128, 128, 32, 0, T.bool(False), 32, 128, 0, 0, 1, 0, 0, 0, 0)
            T.copy(T.region(C_local[0, 0], 1, 128, 128), T.region(C[by * 128, bx * 128], 2, 128, 128))