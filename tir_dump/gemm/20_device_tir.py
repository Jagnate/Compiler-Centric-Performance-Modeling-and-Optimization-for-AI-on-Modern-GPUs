# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def matmul_kernel(A: T.handle("float16", "global"), B: T.handle("float16", "global"), C: T.handle("float16", "global")):
        T.func_attr({"calling_conv": 2, "dyn_shared_memory_buf": 49152, "target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "thread_extent": {"blockIdx.x": 8, "blockIdx.y": 8, "threadIdx.x": 128, "threadIdx.y": 1, "threadIdx.z": 1}, "tir.is_global_func": T.bool(True), "tir.kernel_launch_params": ["blockIdx.x", "blockIdx.y", "threadIdx.x", "threadIdx.y", "threadIdx.z", "tir.use_dyn_shared_memory"], "tir.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1]})
        C_1 = T.decl_buffer((1048576,), "float16", data=C)
        C_local = T.handle("float32", "local")
        C_local_1 = T.decl_buffer((128,), data=C_local, scope="local")
        C_local_cast = T.handle("float16", "local")
        C_local_cast_1 = T.decl_buffer((2,), "float16", data=C_local_cast, scope="local")
        C_local_2 = T.decl_buffer((128,), data=C_local, scope="local")
        bx = T.launch_thread("blockIdx.x", 8)
        buf_dyn_shmem = T.allocate([49152], "uint8", "shared.dyn")
        C_local = T.allocate([128], "float32", "local")
        C_local_cast = T.allocate([2], "float16", "local")
        by = T.launch_thread("blockIdx.y", 8)
        thread_binding = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        for i in T.unroll(32):
            C_local_3 = T.Buffer((128,), data=C_local, scope="local")
            C_local_3[i * 4:i * 4 + 4] = T.Broadcast(T.float32(0.0), 4)
        with T.attr(0, "tl.pipeline_mvb_num_stages", 3):
            with T.attr(0, "async_scope", 1):
                T.attr(0, "tl.pipeline_mvb_stage_expr", 0)
                T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                for i in T.unroll(4):
                    T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, i * 1024 + thread_binding // 4 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), A, by * 131072 + i * 32768 + thread_binding // 4 * 1024 + thread_binding % 4 * 8, 1, 1), 8)
            with T.attr(0, "async_scope", 1):
                T.attr(0, "tl.pipeline_mvb_stage_expr", 0)
                T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                for i in T.unroll(4):
                    T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, thread_binding % 16 // 8 * 2048 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + 12288, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B, i * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8, 1, 1), 8)
            T.ptx_commit_group()
            with T.attr(0, "async_scope", 1):
                T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
                T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                for i in T.unroll(4):
                    T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, i * 1024 + thread_binding // 4 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), A, by * 131072 + i * 32768 + thread_binding // 4 * 1024 + thread_binding % 4 * 8 + 32, 1, 1), 8)
            with T.attr(0, "async_scope", 1):
                T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
                T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                for i in T.unroll(4):
                    T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, thread_binding % 16 // 8 * 2048 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + 16384, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B, i * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8 + 32768, 1, 1), 8)
            T.ptx_commit_group()
            for k in T.serial(30, annotations={"tl_pipelined_num_stages": 3}):
                with T.attr(0, "async_scope", 1):
                    T.attr(0, "tl.pipeline_mvb_stage_expr", (k + 2) % 3)
                    T.attr(0, "tl.pipeline_mvb_parity_expr", (k + 2) % 6 // 3)
                    T.tvm_storage_sync("shared.dyn")
                    for i in T.unroll(4):
                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, (k + 2) % 3 * 4096 + i * 1024 + thread_binding // 4 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), A, by * 131072 + i * 32768 + thread_binding // 4 * 1024 + k * 32 + thread_binding % 4 * 8 + 64, 1, 1), 8)
                with T.attr(0, "async_scope", 1):
                    T.attr(0, "tl.pipeline_mvb_stage_expr", (k + 2) % 3)
                    T.attr(0, "tl.pipeline_mvb_parity_expr", (k + 2) % 6 // 3)
                    for i in T.unroll(4):
                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, (k + 2) % 3 * 4096 + thread_binding % 16 // 8 * 2048 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + 12288, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B, k * 32768 + i * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8 + 65536, 1, 1), 8)
                T.ptx_commit_group()
                T.ptx_wait_group(2)
                T.tvm_storage_sync("shared")
                T.attr(0, "tl.pipeline_mvb_stage_expr", k % 3)
                T.attr(0, "tl.pipeline_mvb_parity_expr", k % 6 // 3)
                T.attr(0, "lexical_alloc_scope", 1)
                A_local = T.allocate([32], "float16", "local")
                B_local = T.allocate([32], "float16", "local")
                for ki in range(2):
                    for i in range(4):
                        T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, k % 3 * 4096 + thread_binding % 64 // 32 * 2048 + i * 512 + thread_binding % 16 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local, i * 8, 8, 2))
                    for i in range(4):
                        T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, k % 3 * 4096 + thread_binding // 64 * 2048 + ki * 1024 + thread_binding % 16 // 8 * 512 + (thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512 + 12288, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local, i * 8, 8, 2))
                    for i, j in T.grid(4, 4):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i * 8, B_local, j * 8, C_local, i * 32 + j * 8, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i * 8, B_local, j * 8 + 4, C_local, i * 32 + j * 8 + 4, T.bool(False))
            T.ptx_wait_group(2)
            T.tvm_storage_sync("shared")
            with T.attr(0, "tl.pipeline_mvb_stage_expr", 0):
                T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                T.attr(0, "lexical_alloc_scope", 1)
                A_local = T.allocate([32], "float16", "local")
                B_local = T.allocate([32], "float16", "local")
                for ki in range(2):
                    for i in range(4):
                        T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, thread_binding % 64 // 32 * 2048 + i * 512 + thread_binding % 16 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local, i * 8, 8, 2))
                    for i in range(4):
                        T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, thread_binding // 64 * 2048 + ki * 1024 + thread_binding % 16 // 8 * 512 + (thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512 + 12288, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local, i * 8, 8, 2))
                    for i, j in T.grid(4, 4):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i * 8, B_local, j * 8, C_local, i * 32 + j * 8, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i * 8, B_local, j * 8 + 4, C_local, i * 32 + j * 8 + 4, T.bool(False))
            T.ptx_wait_group(0)
            T.tvm_storage_sync("shared")
            T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
            T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
            T.attr(0, "lexical_alloc_scope", 1)
            A_local = T.allocate([32], "float16", "local")
            B_local = T.allocate([32], "float16", "local")
            for ki in range(2):
                for i in range(4):
                    T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, thread_binding % 64 // 32 * 2048 + i * 512 + thread_binding % 16 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 8 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local, i * 8, 8, 2))
                for i in range(4):
                    T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, thread_binding // 64 * 2048 + ki * 1024 + thread_binding % 16 // 8 * 512 + (thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512 + 16384, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local, i * 8, 8, 2))
                for i, j in T.grid(4, 4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i * 8, B_local, j * 8, C_local, i * 32 + j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i * 8, B_local, j * 8 + 4, C_local, i * 32 + j * 8 + 4, T.bool(False))
        for i in T.unroll(64):
            C_local_cast_2 = T.Buffer((2,), "float16", data=C_local_cast, scope="local")
            C_local_3 = T.Buffer((128,), data=C_local, scope="local")
            C_local_cast_2[0:2] = T.Cast("float16x2", C_local_3[i * 2:i * 2 + 2])
            C_2 = T.Buffer((1048576,), "float16", data=C)
            C_2[by * 131072 + thread_binding % 64 // 32 * 65536 + i // 16 * 16384 + i % 2 * 8192 + thread_binding % 32 // 4 * 1024 + bx * 128 + thread_binding // 64 * 64 + i % 16 // 2 * 8 + thread_binding % 4 * 2:by * 131072 + thread_binding % 64 // 32 * 65536 + i // 16 * 16384 + i % 2 * 8192 + thread_binding % 32 // 4 * 1024 + bx * 128 + thread_binding // 64 * 64 + i % 16 // 2 * 8 + thread_binding % 4 * 2 + 2] = C_local_cast_2[0:2]