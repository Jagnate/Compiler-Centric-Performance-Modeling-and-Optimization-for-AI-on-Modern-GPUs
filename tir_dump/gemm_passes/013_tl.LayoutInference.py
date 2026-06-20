# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def matmul(A: T.handle, B: T.handle, C_handle: T.handle):
        T.func_attr({"target": T.target({"arch": "sm_86", "host": {"keys": ["cpu"], "kind": "c", "tag": ""}, "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tilelang_out_idx": [-1]})
        A_1 = T.match_buffer(A, (1024, 1024), "float16", strides=(1024, 1))
        B_1 = T.match_buffer(B, (1024, 1024), "float16", strides=(1024, 1))
        C = T.match_buffer(C_handle, (1024, 1024), "float16", strides=(1024, 1))
        with T.block("root"):
            by = T.int32()
            k = T.int32()
            bx = T.int32()
            T.reads(A_1[by * 128:by * 128 + 128, T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64))], B_1[T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64)), bx * 128:bx * 128 + 128], C[by * 128, bx * 128])
            T.writes()
            A_shared = T.Buffer((3, 128, 32), "float16", scope="shared.dyn")
            B_shared = T.Buffer((3, 32, 128), "float16", scope="shared.dyn")
            C_local = T.Buffer((128, 128), scope="local.fragment")
            T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
            T.launch_thread(bx, 8)
            T.launch_thread(by, 8)
            tx = T.launch_thread("threadIdx.x", 128)
            ty = T.launch_thread("threadIdx.y", 1)
            tz = T.launch_thread("threadIdx.z", 1)
            with T.block("tilelang_root"):
                T.reads(A_1[by * 128:by * 128 + 128, T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64))], A_shared[0:3, 0:128, 0:32], B_1[T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64)), bx * 128:bx * 128 + 128], B_shared[0:3, 0:32, 0:128], C[by * 128, bx * 128])
                T.writes(A_shared[0:3, 0:128, 0:32], B_shared[0:3, 0:32, 0:128])
                T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                A_shared = T.alloc_buffer((3, 128, 32), "float16", data=A_shared.data, scope="shared.dyn")
                B_shared = T.alloc_buffer((3, 32, 128), "float16", data=B_shared.data, scope="shared.dyn")
                C_local = T.alloc_buffer((128, 128), data=C_local.data, scope="local.fragment")
                T.fill(T.region(C_local[0, 0], 2, 128, 128), 0)
                with T.block(""):
                    T.reads(A_1[by * 128:by * 128 + 128, T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64))], A_shared[0:3, 0:128, 0:32], B_1[T.min(0, k * 32 + 64):T.min(0, k * 32 + 64) + (T.max(1023, k * 32 + 95) + 1 - T.min(0, k * 32 + 64)), bx * 128:bx * 128 + 128], B_shared[0:3, 0:32, 0:128], C_local[0:128, 0:128], A_1[by * 128:by * 128 + 128, 0:32], B_1[0:32, bx * 128:bx * 128 + 128], A_1[by * 128:by * 128 + 128, 32:64], B_1[32:64, bx * 128:bx * 128 + 128], A_1[by * 128:by * 128 + 128, k * 32 + 64:k * 32 + 64 + 32], B_1[k * 32 + 64:k * 32 + 64 + 32, bx * 128:bx * 128 + 128], A_shared[k % 3, 0:128, 0:32], B_shared[k % 3, 0:32, 0:128], C_local[0:128, 0:128], A_shared[0, 0:128, 0:32], B_shared[0, 0:32, 0:128], C_local[0:128, 0:128], A_shared[1, 0:128, 0:32], B_shared[1, 0:32, 0:128], C_local[0:128, 0:128])
                    T.writes(A_shared[0:3, 0:128, 0:32], B_shared[0:3, 0:32, 0:128], C_local[0:128, 0:128], A_shared[0, 0:128, 0:32], B_shared[0, 0:32, 0:128], A_shared[1, 0:128, 0:32], B_shared[1, 0:32, 0:128], A_shared[(k + 2) % 3, 0:128, 0:32], B_shared[(k + 2) % 3, 0:32, 0:128], C_local[0:128, 0:128], C_local[0:128, 0:128], C_local[0:128, 0:128])
                    T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                    T.attr(0, "tl.pipeline_context_num_stages", 3)
                    T.attr(0, "tl.pipeline_mvb_num_stages", 3)
                    with T.block(""):
                        T.reads(A_1[by * 128:by * 128 + 128, 0:32], A_shared[0:3, 0, 0], B_1[0:32, bx * 128:bx * 128 + 128], B_shared[0:3, 0, 0], A_1[by * 128:by * 128 + 128, 0:32], B_1[0:32, bx * 128:bx * 128 + 128])
                        T.writes(A_shared[0:3, 0:128, 0:32], B_shared[0:3, 0:32, 0:128], A_shared[0, 0:128, 0:32], B_shared[0, 0:32, 0:128])
                        T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                        with T.attr(0, "async_scope", 1):
                            T.attr(0, "tl.pipeline_mvb_stage_expr", 0)
                            T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                            with T.block(""):
                                T.reads(A_1[by * 128, 0], A_shared[0:3, 0, 0], A_1[by * 128:by * 128 + 128, 0:32])
                                T.writes(A_shared[0:3, 0:128, 0:32])
                                T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                                T.copy(T.region(A_1[by * 128, 0], 1, 128, 32), T.region(A_shared[0, 0, 0], 2, 1, 128, 32), no_implicit_async_commit_wait=1, tl.pipeline_mbar_phase_expr=0)
                        with T.attr(0, "async_scope", 1):
                            T.attr(0, "tl.pipeline_mvb_stage_expr", 0)
                            T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                            with T.block(""):
                                T.reads(B_1[0, bx * 128], B_shared[0:3, 0, 0], B_1[0:32, bx * 128:bx * 128 + 128])
                                T.writes(B_shared[0:3, 0:32, 0:128])
                                T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                                T.copy(T.region(B_1[0, bx * 128], 1, 32, 128), T.region(B_shared[0, 0, 0], 2, 1, 32, 128), no_implicit_async_commit_wait=1, tl.pipeline_mbar_phase_expr=0)
                        T.ptx_commit_group()
                    with T.block(""):
                        T.reads(A_1[by * 128:by * 128 + 128, 32:64], A_shared[0:3, 0, 0], B_1[32:64, bx * 128:bx * 128 + 128], B_shared[0:3, 0, 0], A_1[by * 128:by * 128 + 128, 32:64], B_1[32:64, bx * 128:bx * 128 + 128])
                        T.writes(A_shared[0:3, 0:128, 0:32], B_shared[0:3, 0:32, 0:128], A_shared[1, 0:128, 0:32], B_shared[1, 0:32, 0:128])
                        T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                        with T.attr(0, "async_scope", 1):
                            T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
                            T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                            with T.block(""):
                                T.reads(A_1[by * 128, 32], A_shared[0:3, 0, 0], A_1[by * 128:by * 128 + 128, 32:64])
                                T.writes(A_shared[0:3, 0:128, 0:32])
                                T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                                T.copy(T.region(A_1[by * 128, 32], 1, 128, 32), T.region(A_shared[1, 0, 0], 2, 1, 128, 32), no_implicit_async_commit_wait=1, tl.pipeline_mbar_phase_expr=0)
                        with T.attr(0, "async_scope", 1):
                            T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
                            T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                            with T.block(""):
                                T.reads(B_1[32, bx * 128], B_shared[0:3, 0, 0], B_1[32:64, bx * 128:bx * 128 + 128])
                                T.writes(B_shared[0:3, 0:32, 0:128])
                                T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                                T.copy(T.region(B_1[32, bx * 128], 1, 32, 128), T.region(B_shared[1, 0, 0], 2, 1, 32, 128), no_implicit_async_commit_wait=1, tl.pipeline_mbar_phase_expr=0)
                        T.ptx_commit_group()
                    with T.block(""):
                        T.reads(A_1[by * 128:by * 128 + 128, 64:1024], A_shared[0:3, 0:128, 0:32], B_1[64:1024, bx * 128:bx * 128 + 128], B_shared[0:3, 0:32, 0:128], C_local[0:128, 0:128], A_1[by * 128:by * 128 + 128, k * 32 + 64:k * 32 + 64 + 32], B_1[k * 32 + 64:k * 32 + 64 + 32, bx * 128:bx * 128 + 128], A_shared[k % 3, 0:128, 0:32], B_shared[k % 3, 0:32, 0:128], C_local[0:128, 0:128])
                        T.writes(A_shared[0:3, 0:128, 0:32], B_shared[0:3, 0:32, 0:128], C_local[0:128, 0:128], A_shared[(k + 2) % 3, 0:128, 0:32], B_shared[(k + 2) % 3, 0:32, 0:128], C_local[0:128, 0:128])
                        T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                        for k in T.serial(30, annotations={"tl_pipelined_num_stages": 3}):
                            with T.attr(0, "async_scope", 1):
                                T.attr(0, "tl.pipeline_mvb_stage_expr", (k + 2) % 3)
                                T.attr(0, "tl.pipeline_mvb_parity_expr", (k + 2) % 6 // 3)
                                with T.block(""):
                                    T.reads(A_1[by * 128, k * 32 + 64], A_shared[0:3, 0, 0], A_1[by * 128:by * 128 + 128, k * 32 + 64:k * 32 + 64 + 32])
                                    T.writes(A_shared[0:3, 0:128, 0:32])
                                    T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                                    T.copy(T.region(A_1[by * 128, k * 32 + 64], 1, 128, 32), T.region(A_shared[(k + 2) % 3, 0, 0], 2, 1, 128, 32), no_implicit_async_commit_wait=1, tl.pipeline_mbar_phase_expr=(k + 2) % 6 // 3)
                            with T.attr(0, "async_scope", 1):
                                T.attr(0, "tl.pipeline_mvb_stage_expr", (k + 2) % 3)
                                T.attr(0, "tl.pipeline_mvb_parity_expr", (k + 2) % 6 // 3)
                                with T.block(""):
                                    T.reads(B_1[k * 32 + 64, bx * 128], B_shared[0:3, 0, 0], B_1[k * 32 + 64:k * 32 + 64 + 32, bx * 128:bx * 128 + 128])
                                    T.writes(B_shared[0:3, 0:32, 0:128])
                                    T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                                    T.copy(T.region(B_1[k * 32 + 64, bx * 128], 1, 32, 128), T.region(B_shared[(k + 2) % 3, 0, 0], 2, 1, 32, 128), no_implicit_async_commit_wait=1, tl.pipeline_mbar_phase_expr=(k + 2) % 6 // 3)
                            T.ptx_commit_group()
                            T.ptx_wait_group(2)
                            T.attr(0, "tl.pipeline_mvb_stage_expr", k % 3)
                            T.attr(0, "tl.pipeline_mvb_parity_expr", k % 6 // 3)
                            with T.block(""):
                                T.reads(A_shared[0:3, 0, 0], B_shared[0:3, 0, 0], C_local[0, 0], A_shared[0:3, 0:128, 0:32], B_shared[0:3, 0:32, 0:128], C_local[0:128, 0:128])
                                T.writes(C_local[0:128, 0:128])
                                T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                                T.gemm(T.region(A_shared[k % 3, 0, 0], 1, 1, 128, 32), T.region(B_shared[k % 3, 0, 0], 1, 1, 32, 128), T.region(C_local[0, 0], 3, 128, 128), T.bool(False), T.bool(False), 128, 128, 32, 0, T.bool(False), 32, 128, 0, 0, 1, 0, 0, 0, 0, tl.pipeline_mbar_phase_expr=k % 6 // 3)
                    with T.block(""):
                        T.reads(A_shared[0:3, 0:128, 0:32], B_shared[0:3, 0:32, 0:128], C_local[0:128, 0:128], A_shared[0, 0:128, 0:32], B_shared[0, 0:32, 0:128], C_local[0:128, 0:128])
                        T.writes(C_local[0:128, 0:128], C_local[0:128, 0:128])
                        T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                        T.ptx_wait_group(2)
                        T.attr(0, "tl.pipeline_mvb_stage_expr", 0)
                        T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                        with T.block(""):
                            T.reads(A_shared[0:3, 0, 0], B_shared[0:3, 0, 0], C_local[0, 0], A_shared[0:3, 0:128, 0:32], B_shared[0:3, 0:32, 0:128], C_local[0:128, 0:128])
                            T.writes(C_local[0:128, 0:128])
                            T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                            T.gemm(T.region(A_shared[0, 0, 0], 1, 1, 128, 32), T.region(B_shared[0, 0, 0], 1, 1, 32, 128), T.region(C_local[0, 0], 3, 128, 128), T.bool(False), T.bool(False), 128, 128, 32, 0, T.bool(False), 32, 128, 0, 0, 1, 0, 0, 0, 0, tl.pipeline_mbar_phase_expr=0)
                    with T.block(""):
                        T.reads(A_shared[0:3, 0:128, 0:32], B_shared[0:3, 0:32, 0:128], C_local[0:128, 0:128], A_shared[1, 0:128, 0:32], B_shared[1, 0:32, 0:128], C_local[0:128, 0:128])
                        T.writes(C_local[0:128, 0:128], C_local[0:128, 0:128])
                        T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                        T.ptx_wait_group(0)
                        T.attr(0, "tl.pipeline_mvb_stage_expr", 1)
                        T.attr(0, "tl.pipeline_mvb_parity_expr", 0)
                        with T.block(""):
                            T.reads(A_shared[0:3, 0, 0], B_shared[0:3, 0, 0], C_local[0, 0], A_shared[0:3, 0:128, 0:32], B_shared[0:3, 0:32, 0:128], C_local[0:128, 0:128])
                            T.writes(C_local[0:128, 0:128])
                            T.block_attr({"layout_map": {A_shared: metadata["tl.Layout"][0], B_shared: metadata["tl.Layout"][1], C_local: metadata["tl.Fragment"][0]}})
                            T.gemm(T.region(A_shared[1, 0, 0], 1, 1, 128, 32), T.region(B_shared[1, 0, 0], 1, 1, 32, 128), T.region(C_local[0, 0], 3, 128, 128), T.bool(False), T.bool(False), 128, 128, 32, 0, T.bool(False), 32, 128, 0, 0, 1, 0, 0, 0, 0, tl.pipeline_mbar_phase_expr=0)
                T.copy(T.region(C_local[0, 0], 1, 128, 128), T.region(C[by * 128, bx * 128], 2, 128, 128))

# Metadata omitted. Use show_meta=True in script() method to show it.