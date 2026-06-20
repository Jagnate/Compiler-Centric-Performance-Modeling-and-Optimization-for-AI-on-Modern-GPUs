# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def matmul(A: T.handle, B: T.handle, C_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_86", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tilelang_out_idx": [-1], "tl.has_tma": T.bool(False)})
        A_1 = T.match_buffer(A, (1024, 1024), "float16", strides=(1024, 1))
        B_1 = T.match_buffer(B, (1024, 1024), "float16", strides=(1024, 1))
        C = T.match_buffer(C_handle, (1024, 1024), "float16", strides=(1024, 1))
        bx = T.launch_thread("blockIdx.x", 8)
        by = T.launch_thread("blockIdx.y", 8)
        thread_binding = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        C_local = T.allocate([128], "float32", "local")
        C_local = T.allocate([128], "float32", "local")
        i = T.int32()
        with T.attr(i, "pragma_unroll_explicit", T.bool(False)):
            for i in T.unroll(32):
                for vec in T.vectorized(4):
                    C_local_1 = T.Buffer((128,), data=C_local, scope="local")
                    C_local_1[i * 4 + vec] = T.float32(0.0)
        with T.allocate([12288], "float16", "shared.dyn") as A_shared:
            B_shared = T.allocate([12288], "float16", "shared.dyn")
            T.attr(0, "tl.pipeline_mvb_num_stages", 3)
            with T.attr(0, "async_scope", 1):
                T.attr(0, "tl.pipeline_mvb_stage_expr", 0)
                T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                i_1 = T.int32()
                T.attr(i_1, "pragma_unroll_explicit", T.bool(False))
                for i_1 in T.unroll(4):
                    for vec in T.vectorized(8):
                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), A_shared, i_1 * 1024 + thread_binding // 4 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, by * 131072 + i_1 * 32768 + thread_binding // 4 * 1024 + thread_binding % 4 * 8 + vec, 1, 1), 1)
            with T.attr(0, "async_scope", 1):
                T.attr(0, "tl.pipeline_mvb_stage_expr", 0)
                T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                i_1 = T.int32()
                T.attr(i_1, "pragma_unroll_explicit", T.bool(False))
                for i_1 in T.unroll(4):
                    for vec in T.vectorized(8):
                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), B_shared, thread_binding % 16 // 8 * 2048 + i_1 * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B_1.data, i_1 * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8 + vec, 1, 1), 1)
            T.ptx_commit_group()
            with T.attr(0, "async_scope", 1):
                T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
                T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                i_1 = T.int32()
                T.attr(i_1, "pragma_unroll_explicit", T.bool(False))
                for i_1 in T.unroll(4):
                    for vec in T.vectorized(8):
                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), A_shared, i_1 * 1024 + thread_binding // 4 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, by * 131072 + i_1 * 32768 + thread_binding // 4 * 1024 + thread_binding % 4 * 8 + vec + 32, 1, 1), 1)
            with T.attr(0, "async_scope", 1):
                T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
                T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                i_1 = T.int32()
                T.attr(i_1, "pragma_unroll_explicit", T.bool(False))
                for i_1 in T.unroll(4):
                    for vec in T.vectorized(8):
                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), B_shared, thread_binding % 16 // 8 * 2048 + i_1 * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + vec + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B_1.data, i_1 * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8 + vec + 32768, 1, 1), 1)
            T.ptx_commit_group()
            for k in T.serial(30, annotations={"tl_pipelined_num_stages": 3}):
                with T.attr(0, "async_scope", 1):
                    T.attr(0, "tl.pipeline_mvb_stage_expr", (k + 2) % 3)
                    T.attr(0, "tl.pipeline_mvb_parity_expr", (k + 2) % 6 // 3)
                    i_1 = T.int32()
                    T.attr(i_1, "pragma_unroll_explicit", T.bool(False))
                    for i_1 in T.unroll(4):
                        for vec in T.vectorized(8):
                            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), A_shared, (k + 2) % 3 * 4096 + i_1 * 1024 + thread_binding // 4 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, by * 131072 + i_1 * 32768 + thread_binding // 4 * 1024 + k * 32 + thread_binding % 4 * 8 + vec + 64, 1, 1), 1)
                with T.attr(0, "async_scope", 1):
                    T.attr(0, "tl.pipeline_mvb_stage_expr", (k + 2) % 3)
                    T.attr(0, "tl.pipeline_mvb_parity_expr", (k + 2) % 6 // 3)
                    i_1 = T.int32()
                    T.attr(i_1, "pragma_unroll_explicit", T.bool(False))
                    for i_1 in T.unroll(4):
                        for vec in T.vectorized(8):
                            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), B_shared, (k + 2) % 3 * 4096 + thread_binding % 16 // 8 * 2048 + i_1 * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B_1.data, k * 32768 + i_1 * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8 + vec + 65536, 1, 1), 1)
                T.ptx_commit_group()
                T.ptx_wait_group(2)
                T.attr(0, "tl.pipeline_mvb_stage_expr", k % 3)
                T.attr(0, "tl.pipeline_mvb_parity_expr", k % 6 // 3)
                T.attr(0, "lexical_alloc_scope", 1)
                for ki in range(2):
                    A_local = T.allocate([32], "float16", "local")
                    B_local = T.allocate([32], "float16", "local")
                    for i_1 in range(4):
                        T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, k % 3 * 4096 + thread_binding % 64 // 32 * 2048 + i_1 * 512 + (thread_binding % 16 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16) // 256 * 256 + thread_binding % 8 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local, i_1 * 8, 8, 2))
                    for i_1 in range(4):
                        T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), B_shared, k % 3 * 4096 + thread_binding // 64 * 2048 + ki * 1024 + (thread_binding % 16 * 64 + ((i_1 * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32) // 512 * 512 + (thread_binding % 16 * 64 + ((i_1 * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + i_1 % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local, i_1 * 8, 8, 2))
                    for i_1, j in T.grid(4, 4):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i_1 * 8, B_local, j * 8, C_local, i_1 * 32 + j * 8, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i_1 * 8, B_local, j * 8 + 4, C_local, i_1 * 32 + j * 8 + 4, T.bool(False))
            T.ptx_wait_group(2)
            with T.attr(0, "tl.pipeline_mvb_stage_expr", 0):
                T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                T.attr(0, "lexical_alloc_scope", 1)
                for ki in range(2):
                    A_local = T.allocate([32], "float16", "local")
                    B_local = T.allocate([32], "float16", "local")
                    for i_1 in range(4):
                        T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, thread_binding % 64 // 32 * 2048 + i_1 * 512 + (thread_binding % 16 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16) // 256 * 256 + thread_binding % 8 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local, i_1 * 8, 8, 2))
                    for i_1 in range(4):
                        T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), B_shared, thread_binding // 64 * 2048 + ki * 1024 + (thread_binding % 16 * 64 + ((i_1 * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32) // 512 * 512 + (thread_binding % 16 * 64 + ((i_1 * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + i_1 % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local, i_1 * 8, 8, 2))
                    for i_1, j in T.grid(4, 4):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i_1 * 8, B_local, j * 8, C_local, i_1 * 32 + j * 8, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i_1 * 8, B_local, j * 8 + 4, C_local, i_1 * 32 + j * 8 + 4, T.bool(False))
            T.ptx_wait_group(0)
            T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
            T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
            T.attr(0, "lexical_alloc_scope", 1)
            for ki in range(2):
                A_local = T.allocate([32], "float16", "local")
                B_local = T.allocate([32], "float16", "local")
                for i_1 in range(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), A_shared, thread_binding % 64 // 32 * 2048 + i_1 * 512 + (thread_binding % 16 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16) // 256 * 256 + thread_binding % 8 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 8 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local, i_1 * 8, 8, 2))
                for i_1 in range(4):
                    T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), B_shared, thread_binding // 64 * 2048 + ki * 1024 + (thread_binding % 16 * 64 + ((i_1 * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32) // 512 * 512 + (thread_binding % 16 * 64 + ((i_1 * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + i_1 % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local, i_1 * 8, 8, 2))
                for i_1, j in T.grid(4, 4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i_1 * 8, B_local, j * 8, C_local, i_1 * 32 + j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i_1 * 8, B_local, j * 8 + 4, C_local, i_1 * 32 + j * 8 + 4, T.bool(False))
        i_1 = T.int32()
        T.attr(i_1, "pragma_unroll_explicit", T.bool(False))
        for i_1 in T.unroll(64):
            C_local_cast = T.allocate([2], "float16", "local")
            C_local_cast_1 = T.Buffer((2,), "float16", data=C_local_cast, scope="local")
            for vec in T.vectorized(2):
                C_local_1 = T.Buffer((128,), data=C_local, scope="local")
                C_local_cast_1[vec] = T.Cast("float16", C_local_1[i_1 * 2 + vec])
            for vec_copy in T.vectorized(2):
                C_1 = T.Buffer((1048576,), "float16", data=C.data)
                C_1[by * 131072 + thread_binding % 64 // 32 * 65536 + i_1 // 16 * 16384 + i_1 % 2 * 8192 + thread_binding % 32 // 4 * 1024 + bx * 128 + thread_binding // 64 * 64 + i_1 % 16 // 2 * 8 + thread_binding % 4 * 2 + vec_copy] = C_local_cast_1[vec_copy]