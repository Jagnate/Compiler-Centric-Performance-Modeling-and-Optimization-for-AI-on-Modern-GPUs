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
        with T.block("root"):
            by = T.int32()
            k = T.int32()
            bx = T.int32()
            T.reads(A_1[by * 128:by * 128 + 128, T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64))], B_1[T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64)), bx * 128:bx * 128 + 128], C[by * 128, bx * 128])
            T.writes()
            C_local = T.Buffer((128,), scope="local")
            A_shared = T.Buffer((3, 1, 16, 256), "float16", scope="shared.dyn")
            B_shared = T.Buffer((3, 2, 4, 512), "float16", scope="shared.dyn")
            A_shared_1 = T.Buffer((3, 1, 16, 256), "float16", data=A_shared.data, scope="shared.dyn")
            B_shared_1 = T.Buffer((3, 2, 4, 512), "float16", data=B_shared.data, scope="shared.dyn")
            A_shared_2 = T.Buffer((3, 1, 16, 256), "float16", data=A_shared.data, scope="shared.dyn")
            B_shared_2 = T.Buffer((3, 2, 4, 512), "float16", data=B_shared.data, scope="shared.dyn")
            A_shared_3 = T.Buffer((3, 1, 16, 256), "float16", data=A_shared.data, scope="shared.dyn")
            B_shared_3 = T.Buffer((3, 2, 4, 512), "float16", data=B_shared.data, scope="shared.dyn")
            A_shared_4 = T.Buffer((3, 1, 16, 256), "float16", data=A_shared.data, scope="shared.dyn")
            B_shared_4 = T.Buffer((3, 2, 4, 512), "float16", data=B_shared.data, scope="shared.dyn")
            A_shared_5 = T.Buffer((3, 1, 16, 256), "float16", data=A_shared.data, scope="shared.dyn")
            B_shared_5 = T.Buffer((3, 2, 4, 512), "float16", data=B_shared.data, scope="shared.dyn")
            C_local_1 = T.Buffer((128,), data=C_local.data, scope="local")
            T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
            T.launch_thread(bx, 8)
            T.launch_thread(by, 8)
            thread_binding = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.block("tilelang_root"):
                T.reads(A_1[by * 128:by * 128 + 128, T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64))], B_1[T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64)), bx * 128:bx * 128 + 128], C[by * 128, bx * 128])
                T.writes()
                T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                C_local_1 = T.alloc_buffer((128,), data=C_local.data, scope="local")
                C_local = T.alloc_buffer((128,), data=C_local.data, scope="local")
                for i in T.unroll(32, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    for vec in T.vectorized(4):
                        C_local[i * 4 + vec] = T.float32(0.0)
                with T.block(""):
                    T.reads(A_1[by * 128:by * 128 + 128, T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64))], B_1[T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64)), bx * 128:bx * 128 + 128], A_1[by * 128:by * 128 + 128, 0:32], B_1[0:32, bx * 128:bx * 128 + 128], A_1[by * 128:by * 128 + 128, 32:64], B_1[32:64, bx * 128:bx * 128 + 128], A_1[by * 128:by * 128 + 128, k * 32 + 64:k * 32 + 64 + 32], B_1[k * 32 + 64:k * 32 + 64 + 32, bx * 128:bx * 128 + 128])
                    T.writes()
                    T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                    A_shared_5 = T.alloc_buffer((3, 1, 16, 256), "float16", data=A_shared.data, scope="shared.dyn")
                    B_shared_5 = T.alloc_buffer((3, 2, 4, 512), "float16", data=B_shared.data, scope="shared.dyn")
                    T.attr(0, "tl.pipeline_mvb_num_stages", 3)
                    with T.block(""):
                        T.reads(A_1[by * 128:by * 128 + 128, 0:32], B_1[0:32, bx * 128:bx * 128 + 128], A_1[by * 128:by * 128 + 128, 0:32], B_1[0:32, bx * 128:bx * 128 + 128])
                        T.writes()
                        T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                        with T.attr(0, "async_scope", 1):
                            T.attr(0, "tl.pipeline_mvb_stage_expr", 0)
                            T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                            with T.block(""):
                                T.reads(A_1[by * 128, 0], A_1[by * 128:by * 128 + 128, 0:32])
                                T.writes()
                                T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                    for vec in T.vectorized(8):
                                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, i * 1024 + thread_binding // 4 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, by * 131072 + i * 32768 + thread_binding // 4 * 1024 + thread_binding % 4 * 8 + vec, 1, 1), 1)
                        with T.attr(0, "async_scope", 1):
                            T.attr(0, "tl.pipeline_mvb_stage_expr", 0)
                            T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                            with T.block(""):
                                T.reads(B_1[0, bx * 128], B_1[0:32, bx * 128:bx * 128 + 128])
                                T.writes()
                                T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                    for vec in T.vectorized(8):
                                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, thread_binding % 16 // 8 * 2048 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B_1.data, i * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8 + vec, 1, 1), 1)
                        T.ptx_commit_group()
                    with T.block(""):
                        T.reads(A_1[by * 128:by * 128 + 128, 32:64], B_1[32:64, bx * 128:bx * 128 + 128], A_1[by * 128:by * 128 + 128, 32:64], B_1[32:64, bx * 128:bx * 128 + 128])
                        T.writes()
                        T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                        with T.attr(0, "async_scope", 1):
                            T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
                            T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                            with T.block(""):
                                T.reads(A_1[by * 128, 32], A_1[by * 128:by * 128 + 128, 32:64])
                                T.writes()
                                T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                    for vec in T.vectorized(8):
                                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, i * 1024 + thread_binding // 4 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, by * 131072 + i * 32768 + thread_binding // 4 * 1024 + thread_binding % 4 * 8 + vec + 32, 1, 1), 1)
                        with T.attr(0, "async_scope", 1):
                            T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
                            T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                            with T.block(""):
                                T.reads(B_1[32, bx * 128], B_1[32:64, bx * 128:bx * 128 + 128])
                                T.writes()
                                T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                                for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                    for vec in T.vectorized(8):
                                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, thread_binding % 16 // 8 * 2048 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + vec + 4096, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B_1.data, i * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8 + vec + 32768, 1, 1), 1)
                        T.ptx_commit_group()
                    with T.block(""):
                        T.reads(A_1[by * 128:by * 128 + 128, 64:1024], B_1[64:1024, bx * 128:bx * 128 + 128], A_1[by * 128:by * 128 + 128, k * 32 + 64:k * 32 + 64 + 32], B_1[k * 32 + 64:k * 32 + 64 + 32, bx * 128:bx * 128 + 128])
                        T.writes()
                        T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                        for k in T.serial(30, annotations={"tl_pipelined_num_stages": 3}):
                            with T.attr(0, "async_scope", 1):
                                T.attr(0, "tl.pipeline_mvb_stage_expr", (k + 2) % 3)
                                T.attr(0, "tl.pipeline_mvb_parity_expr", (k + 2) % 6 // 3)
                                with T.block(""):
                                    T.reads(A_1[by * 128, k * 32 + 64], A_1[by * 128:by * 128 + 128, k * 32 + 64:k * 32 + 64 + 32])
                                    T.writes()
                                    T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                        for vec in T.vectorized(8):
                                            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, (k + 2) % 3 * 4096 + i * 1024 + thread_binding // 4 * 32 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 16 // 8 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, by * 131072 + i * 32768 + thread_binding // 4 * 1024 + k * 32 + thread_binding % 4 * 8 + vec + 64, 1, 1), 1)
                            with T.attr(0, "async_scope", 1):
                                T.attr(0, "tl.pipeline_mvb_stage_expr", (k + 2) % 3)
                                T.attr(0, "tl.pipeline_mvb_parity_expr", (k + 2) % 6 // 3)
                                with T.block(""):
                                    T.reads(B_1[k * 32 + 64, bx * 128], B_1[k * 32 + 64:k * 32 + 64 + 32, bx * 128:bx * 128 + 128])
                                    T.writes()
                                    T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                                    for i in T.unroll(4, annotations={"pragma_unroll_explicit": T.bool(False)}):
                                        for vec in T.vectorized(8):
                                            T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, (k + 2) % 3 * 4096 + thread_binding % 16 // 8 * 2048 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + vec, 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B_1.data, k * 32768 + i * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8 + vec + 65536, 1, 1), 1)
                            T.ptx_commit_group()
                            T.ptx_wait_group(2)
                            T.attr(0, "tl.pipeline_mvb_stage_expr", k % 3)
                            T.attr(0, "tl.pipeline_mvb_parity_expr", k % 6 // 3)
                            with T.block(""):
                                T.reads()
                                T.writes()
                                T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                                with T.block("_gemm_ssr"):
                                    T.reads()
                                    T.writes()
                                    T.block_attr({"lexical_alloc_scope": 1})
                                    for ki in range(2):
                                        with T.block(""):
                                            T.reads(A_shared_5[0:3, 0, 0:16, 0:256], B_shared_5[0:3, 0:2, 0:4, 0:512], C_local[0:128])
                                            T.writes(C_local[0:128])
                                            A_local = T.alloc_buffer((32,), "float16", scope="local")
                                            B_local = T.alloc_buffer((32,), "float16", scope="local")
                                            for i in range(4):
                                                T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, k % 3 * 4096 + thread_binding % 64 // 32 * 2048 + i * 512 + (thread_binding % 16 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16) // 256 * 256 + thread_binding % 8 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, i * 8, 8, 2))
                                            for i in range(4):
                                                T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, k % 3 * 4096 + thread_binding // 64 * 2048 + ki * 1024 + (thread_binding % 16 * 64 + ((i * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32) // 512 * 512 + (thread_binding % 16 * 64 + ((i * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                                            for i, j in T.grid(4, 4):
                                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, i * 8, B_local.data, j * 8, C_local.data, i * 32 + j * 8, T.bool(False))
                                                T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, i * 8, B_local.data, j * 8 + 4, C_local.data, i * 32 + j * 8 + 4, T.bool(False))
                    with T.block(""):
                        T.reads()
                        T.writes()
                        T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                        T.ptx_wait_group(2)
                        T.attr(0, "tl.pipeline_mvb_stage_expr", 0)
                        T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                        with T.block(""):
                            T.reads()
                            T.writes()
                            T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                            with T.block("_gemm_ssr"):
                                T.reads()
                                T.writes()
                                T.block_attr({"lexical_alloc_scope": 1})
                                for ki in range(2):
                                    with T.block(""):
                                        T.reads(A_shared_5[0:3, 0, 0:16, 0:256], B_shared_5[0:3, 0:2, 0:4, 0:512], C_local[0:128])
                                        T.writes(C_local[0:128])
                                        A_local = T.alloc_buffer((32,), "float16", scope="local")
                                        B_local = T.alloc_buffer((32,), "float16", scope="local")
                                        for i in range(4):
                                            T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, thread_binding % 64 // 32 * 2048 + i * 512 + (thread_binding % 16 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16) // 256 * 256 + thread_binding % 8 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 8, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, i * 8, 8, 2))
                                        for i in range(4):
                                            T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, thread_binding // 64 * 2048 + ki * 1024 + (thread_binding % 16 * 64 + ((i * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32) // 512 * 512 + (thread_binding % 16 * 64 + ((i * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                                        for i, j in T.grid(4, 4):
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, i * 8, B_local.data, j * 8, C_local.data, i * 32 + j * 8, T.bool(False))
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, i * 8, B_local.data, j * 8 + 4, C_local.data, i * 32 + j * 8 + 4, T.bool(False))
                    with T.block(""):
                        T.reads()
                        T.writes()
                        T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                        T.ptx_wait_group(0)
                        T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
                        T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                        with T.block(""):
                            T.reads()
                            T.writes()
                            T.block_attr({"layout_map": {C_local: metadata["tl.Fragment"][0], A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], A_shared_1: metadata["tl.Layout"][0], B_shared_1: metadata["tl.Layout"][1], A_shared_2: metadata["tl.Layout"][0], B_shared_2: metadata["tl.Layout"][1], A_shared_3: metadata["tl.Layout"][0], B_shared_3: metadata["tl.Layout"][1], A_shared_4: metadata["tl.Layout"][0], B_shared_4: metadata["tl.Layout"][1], A_shared_5: metadata["tl.Layout"][0], B_shared_5: metadata["tl.Layout"][1], C_local_1: metadata["tl.Fragment"][0]}})
                            with T.block("_gemm_ssr"):
                                T.reads()
                                T.writes()
                                T.block_attr({"lexical_alloc_scope": 1})
                                for ki in range(2):
                                    with T.block(""):
                                        T.reads(A_shared_5[0:3, 0, 0:16, 0:256], B_shared_5[0:3, 0:2, 0:4, 0:512], C_local[0:128])
                                        T.writes(C_local[0:128])
                                        A_local = T.alloc_buffer((32,), "float16", scope="local")
                                        B_local = T.alloc_buffer((32,), "float16", scope="local")
                                        for i in range(4):
                                            T.ptx_ldmatrix(T.bool(False), 4, T.tvm_access_ptr(T.type_annotation("float16"), A_shared.data, thread_binding % 64 // 32 * 2048 + i * 512 + (thread_binding % 16 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16) // 256 * 256 + thread_binding % 8 * 32 + (thread_binding % 8 // 4 + ki) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 4 // 2) % 2 * 8 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), A_local.data, i * 8, 8, 2))
                                        for i in range(4):
                                            T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), B_shared.data, thread_binding // 64 * 2048 + ki * 1024 + (thread_binding % 16 * 64 + ((i * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32) // 512 * 512 + (thread_binding % 16 * 64 + ((i * 16 + thread_binding % 32 // 16 * 8) // 32 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512 + 4096, 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local.data, i * 8, 8, 2))
                                        for i, j in T.grid(4, 4):
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, i * 8, B_local.data, j * 8, C_local.data, i * 32 + j * 8, T.bool(False))
                                            T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, i * 8, B_local.data, j * 8 + 4, C_local.data, i * 32 + j * 8 + 4, T.bool(False))
                for i in T.unroll(64, annotations={"pragma_unroll_explicit": T.bool(False)}):
                    C_local_cast = T.decl_buffer((2,), "float16", scope="local")
                    for vec in T.vectorized(2):
                        C_local_cast[vec] = T.Cast("float16", C_local_1[i * 2 + vec])
                    for vec_copy in T.vectorized(2):
                        C[by * 128 + thread_binding % 64 // 32 * 64 + i // 16 * 16 + i % 2 * 8 + thread_binding % 32 // 4, bx * 128 + thread_binding // 64 * 64 + i % 16 // 2 * 8 + thread_binding % 4 * 2 + vec_copy] = C_local_cast[vec_copy]

# Metadata omitted. Use show_meta=True in script() method to show it.