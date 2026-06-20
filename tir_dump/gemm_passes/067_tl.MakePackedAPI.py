# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def matmul(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle("void", "global")) -> T.int32:
        T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_matmul", "target": T.target({"keys": ["cpu"], "kind": "c", "tag": ""}), "tilelang_out_idx": [-1], "tir.is_entry_func": True, "tl.has_tma": T.bool(False), "tl.readonly_param_indices": [0, 1, 2], "tma_descriptor_args": {}})
        assert num_args == 3, "matmul: num_args should be 3"
        assert not T.isnullptr(args), "matmul: args pointer is NULL"
        A_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
        assert A_type_index == 0 or A_type_index == 4 or A_type_index == 7 or A_type_index >= 64, "kernel matmul input A expected pointer or tensor handle"
        B_type_index: T.int32 = T.tvm_struct_get(args, 1, 13, "int32")
        assert B_type_index == 0 or B_type_index == 4 or B_type_index == 7 or B_type_index >= 64, "kernel matmul input B expected pointer or tensor handle"
        C_handle_type_index: T.int32 = T.tvm_struct_get(args, 2, 13, "int32")
        assert C_handle_type_index == 0 or C_handle_type_index == 4 or C_handle_type_index == 7 or C_handle_type_index >= 64, "kernel matmul input C expected pointer or tensor handle"
        A: T.handle = T.Select(A_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 0, 15, "handle"), 24), T.tvm_struct_get(args, 0, 15, "handle"))
        B: T.handle = T.Select(B_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 1, 15, "handle"), 24), T.tvm_struct_get(args, 1, 15, "handle"))
        C_handle: T.handle = T.Select(C_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 2, 15, "handle"), 24), T.tvm_struct_get(args, 2, 15, "handle"))
        matmul_A_is_null: T.bool = T.isnullptr(A)
        assert not matmul_A_is_null, "matmul.A is expected to have non-NULL pointer"
        matmul_B_is_null: T.bool = T.isnullptr(B)
        assert not matmul_B_is_null, "matmul.B is expected to have non-NULL pointer"
        matmul_C_is_null: T.bool = T.isnullptr(C_handle)
        assert not matmul_C_is_null, "matmul.C is expected to have non-NULL pointer"
        matmul_A_shape: T.handle("int64", "global") = T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 2, "handle"), T.reinterpret("handle", T.uint64(0)))
        matmul_A_shape_1 = T.decl_buffer((2,), "int64", data=matmul_A_shape)
        matmul_B_shape: T.handle("int64", "global") = T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 2, "handle"), T.reinterpret("handle", T.uint64(0)))
        matmul_B_shape_1 = T.decl_buffer((2,), "int64", data=matmul_B_shape)
        matmul_C_shape: T.handle("int64", "global") = T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 2, "handle"), T.reinterpret("handle", T.uint64(0)))
        matmul_C_shape_1 = T.decl_buffer((2,), "int64", data=matmul_C_shape)
        if not T.bool(False):
            if not 2 == T.tvm_struct_get(A, 0, 4, "int32"):
                T.call_packed("__tvm_error_ndim_mismatch", "matmul", "A", T.int64(2), T.Cast("int64", T.tvm_struct_get(A, 0, 4, "int32")))
        else:
            T.evaluate(0)
        matmul_A_strides: T.handle("int64", "global") = T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 3, "handle"), T.reinterpret("handle", T.uint64(0)))
        matmul_A_strides_1 = T.decl_buffer((2,), "int64", data=matmul_A_strides)
        dev_id: T.int32 = T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 9, "int32"), 0)
        A_1: T.handle("float16", "global") = T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 1, "handle"), T.reinterpret("handle", T.uint64(0)))
        T.attr(A_1, "storage_alignment", 64)
        if not T.bool(False):
            if not 2 == T.tvm_struct_get(B, 0, 4, "int32"):
                T.call_packed("__tvm_error_ndim_mismatch", "matmul", "B", T.int64(2), T.Cast("int64", T.tvm_struct_get(B, 0, 4, "int32")))
        else:
            T.evaluate(0)
        matmul_B_strides: T.handle("int64", "global") = T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 3, "handle"), T.reinterpret("handle", T.uint64(0)))
        matmul_B_strides_1 = T.decl_buffer((2,), "int64", data=matmul_B_strides)
        B_1: T.handle("float16", "global") = T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 1, "handle"), T.reinterpret("handle", T.uint64(0)))
        T.attr(B_1, "storage_alignment", 64)
        if not T.bool(False):
            if not 2 == T.tvm_struct_get(C_handle, 0, 4, "int32"):
                T.call_packed("__tvm_error_ndim_mismatch", "matmul", "C", T.int64(2), T.Cast("int64", T.tvm_struct_get(C_handle, 0, 4, "int32")))
        else:
            T.evaluate(0)
        matmul_C_strides: T.handle("int64", "global") = T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 3, "handle"), T.reinterpret("handle", T.uint64(0)))
        matmul_C_strides_1 = T.decl_buffer((2,), "int64", data=matmul_C_strides)
        C: T.handle("float16", "global") = T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 1, "handle"), T.reinterpret("handle", T.uint64(0)))
        T.attr(C, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        if not T.bool(False) and not (T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 5, "uint8"), T.uint8(2)) == T.uint8(2) and T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 6, "uint8"), T.uint8(16)) == T.uint8(16) and T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 7, "uint16"), T.uint16(1)) == T.uint16(1)):
            T.call_packed("__tvm_error_dtype_mismatch", "matmul", "A", T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 5, "uint8"), T.uint8(2))), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 6, "uint8"), T.uint8(16))), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 7, "uint16"), T.uint16(1))), T.int64(2), T.int64(16), T.int64(1))
        if not T.bool(False):
            if not T.Cast("int32", matmul_A_shape_1[0]) == 1024:
                T.call_packed("__tvm_error_expect_eq", "matmul", "A", "shape[0]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_A_shape_1[0])))
            if not T.Cast("int32", matmul_A_shape_1[1]) == 1024:
                T.call_packed("__tvm_error_expect_eq", "matmul", "A", "shape[1]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_A_shape_1[1])))
            if not T.if_then_else(T.isnullptr(matmul_A_strides), 1, T.Cast("int32", matmul_A_strides_1[1])) == 1:
                T.call_packed("__tvm_error_expect_eq", "matmul", "A", "strides[1]", T.int64(1), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_A_strides), 1, T.Cast("int32", matmul_A_strides_1[1]))))
            if not T.if_then_else(T.isnullptr(matmul_A_strides), 1, T.Cast("int32", matmul_A_strides_1[0])) == 1024:
                T.call_packed("__tvm_error_expect_eq", "matmul", "A", "strides[0]", T.int64(1024), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_A_strides), 1, T.Cast("int32", matmul_A_strides_1[0]))))
        if not T.bool(False):
            if not T.uint64(0) == T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 8, "uint64"), T.uint64(0)):
                T.call_packed("__tvm_error_byte_offset_mismatch", "matmul", "A", T.int64(0), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 8, "uint64"), T.uint64(0))))
        else:
            T.evaluate(0)
        if not T.bool(False):
            if not 2 == T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 10, "int32"), 0):
                T.call_packed("__tvm_error_device_type_mismatch", "matmul", "A", T.int64(2), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(A, 0, 10, "int32"), 0)))
        else:
            T.evaluate(0)
        if not T.bool(False):
            if not T.bool(False):
                if T.isnullptr(A_1):
                    T.call_packed("__tvm_error_null_ptr", "matmul", "A", "data pointer")
            else:
                T.evaluate(0)
        else:
            T.evaluate(0)
        if not T.bool(False) and not (T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 5, "uint8"), T.uint8(2)) == T.uint8(2) and T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 6, "uint8"), T.uint8(16)) == T.uint8(16) and T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 7, "uint16"), T.uint16(1)) == T.uint16(1)):
            T.call_packed("__tvm_error_dtype_mismatch", "matmul", "B", T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 5, "uint8"), T.uint8(2))), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 6, "uint8"), T.uint8(16))), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 7, "uint16"), T.uint16(1))), T.int64(2), T.int64(16), T.int64(1))
        if not T.bool(False):
            if not T.Cast("int32", matmul_B_shape_1[0]) == 1024:
                T.call_packed("__tvm_error_expect_eq", "matmul", "B", "shape[0]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_B_shape_1[0])))
            if not T.Cast("int32", matmul_B_shape_1[1]) == 1024:
                T.call_packed("__tvm_error_expect_eq", "matmul", "B", "shape[1]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_B_shape_1[1])))
            if not T.if_then_else(T.isnullptr(matmul_B_strides), 1, T.Cast("int32", matmul_B_strides_1[1])) == 1:
                T.call_packed("__tvm_error_expect_eq", "matmul", "B", "strides[1]", T.int64(1), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_B_strides), 1, T.Cast("int32", matmul_B_strides_1[1]))))
            if not T.if_then_else(T.isnullptr(matmul_B_strides), 1, T.Cast("int32", matmul_B_strides_1[0])) == 1024:
                T.call_packed("__tvm_error_expect_eq", "matmul", "B", "strides[0]", T.int64(1024), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_B_strides), 1, T.Cast("int32", matmul_B_strides_1[0]))))
        if not T.bool(False):
            if not T.uint64(0) == T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 8, "uint64"), T.uint64(0)):
                T.call_packed("__tvm_error_byte_offset_mismatch", "matmul", "B", T.int64(0), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 8, "uint64"), T.uint64(0))))
        else:
            T.evaluate(0)
        if not T.bool(False):
            if not T.tvm_struct_get(B, 0, 9, "int32") == T.tvm_struct_get(A, 0, 9, "int32"):
                T.call_packed("__tvm_error_expect_eq", "matmul", "B", "device_id", T.Cast("int64", T.tvm_struct_get(A, 0, 9, "int32")), T.Cast("int64", T.tvm_struct_get(B, 0, 9, "int32")))
        if not T.bool(False):
            if not 2 == T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 10, "int32"), 0):
                T.call_packed("__tvm_error_device_type_mismatch", "matmul", "B", T.int64(2), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(B, 0, 10, "int32"), 0)))
        else:
            T.evaluate(0)
        if not T.bool(False):
            if not T.bool(False):
                if T.isnullptr(B_1):
                    T.call_packed("__tvm_error_null_ptr", "matmul", "B", "data pointer")
            else:
                T.evaluate(0)
        else:
            T.evaluate(0)
        if not T.bool(False) and not (T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 5, "uint8"), T.uint8(2)) == T.uint8(2) and T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 6, "uint8"), T.uint8(16)) == T.uint8(16) and T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 7, "uint16"), T.uint16(1)) == T.uint16(1)):
            T.call_packed("__tvm_error_dtype_mismatch", "matmul", "C", T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 5, "uint8"), T.uint8(2))), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 6, "uint8"), T.uint8(16))), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 7, "uint16"), T.uint16(1))), T.int64(2), T.int64(16), T.int64(1))
        if not T.bool(False):
            if not T.Cast("int32", matmul_C_shape_1[0]) == 1024:
                T.call_packed("__tvm_error_expect_eq", "matmul", "C", "shape[0]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_C_shape_1[0])))
            if not T.Cast("int32", matmul_C_shape_1[1]) == 1024:
                T.call_packed("__tvm_error_expect_eq", "matmul", "C", "shape[1]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_C_shape_1[1])))
            if not T.if_then_else(T.isnullptr(matmul_C_strides), 1, T.Cast("int32", matmul_C_strides_1[1])) == 1:
                T.call_packed("__tvm_error_expect_eq", "matmul", "C", "strides[1]", T.int64(1), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_C_strides), 1, T.Cast("int32", matmul_C_strides_1[1]))))
            if not T.if_then_else(T.isnullptr(matmul_C_strides), 1, T.Cast("int32", matmul_C_strides_1[0])) == 1024:
                T.call_packed("__tvm_error_expect_eq", "matmul", "C", "strides[0]", T.int64(1024), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_C_strides), 1, T.Cast("int32", matmul_C_strides_1[0]))))
        if not T.bool(False):
            if not T.uint64(0) == T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 8, "uint64"), T.uint64(0)):
                T.call_packed("__tvm_error_byte_offset_mismatch", "matmul", "C", T.int64(0), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 8, "uint64"), T.uint64(0))))
        else:
            T.evaluate(0)
        if not T.bool(False):
            if not T.tvm_struct_get(C_handle, 0, 9, "int32") == T.tvm_struct_get(A, 0, 9, "int32"):
                T.call_packed("__tvm_error_expect_eq", "matmul", "C", "device_id", T.Cast("int64", T.tvm_struct_get(A, 0, 9, "int32")), T.Cast("int64", T.tvm_struct_get(C_handle, 0, 9, "int32")))
        if not T.bool(False):
            if not 2 == T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 10, "int32"), 0):
                T.call_packed("__tvm_error_device_type_mismatch", "matmul", "C", T.int64(2), T.Cast("int64", T.if_then_else(not T.bool(False), T.tvm_struct_get(C_handle, 0, 10, "int32"), 0)))
        else:
            T.evaluate(0)
        if not T.bool(False):
            if not T.bool(False):
                if T.isnullptr(C):
                    T.call_packed("__tvm_error_null_ptr", "matmul", "C", "data pointer")
            else:
                T.evaluate(0)
        else:
            T.evaluate(0)
        A_2 = T.decl_buffer((1024, 1024), "float16", data=A_1, strides=(1024, 1))
        B_2 = T.decl_buffer((1024, 1024), "float16", data=B_1, strides=(1024, 1))
        C_1 = T.decl_buffer((1024, 1024), "float16", data=C, strides=(1024, 1))
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "matmul_compute_"):
            Module.matmul_kernel(A_1, B_1, C)
        return 0

    @T.prim_func(private=True)
    def matmul_kernel(A: T.handle("float16", "global"), B: T.handle("float16", "global"), C: T.handle("float16", "global")):
        T.func_attr({"target": T.target({"arch": "sm_86", "keys": ["cuda", "gpu"], "kind": "cuda", "max_num_threads": 1024, "tag": "", "thread_warp_size": 32}), "tir.is_global_func": True, "tir.noalias": True, "tl.non_restrict_params": [], "tl.readonly_param_indices": [0, 1]})
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
                    T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + (thread_binding % 16 // 8 * 2048 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8), 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B, i * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8, 1, 1), 8)
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
                    T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + (thread_binding % 16 // 8 * 2048 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8 + 4096), 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B, i * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8 + 32768, 1, 1), 8)
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
                        T.ptx_cp_async(T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + ((k + 2) % 3 * 4096 + thread_binding % 16 // 8 * 2048 + i * 512 + thread_binding // 16 * 64 + (thread_binding // 64 + thread_binding % 8 // 4) % 2 * 32 + (thread_binding % 64 // 32 + thread_binding % 4 // 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8), 1, 2), T.tvm_access_ptr(T.type_annotation("float16"), B, k * 32768 + i * 8192 + thread_binding // 16 * 1024 + bx * 128 + thread_binding % 16 * 8 + 65536, 1, 1), 8)
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
                        T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + (k % 3 * 4096 + thread_binding // 64 * 2048 + ki * 1024 + thread_binding % 16 // 8 * 512 + (thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512), 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local, i * 8, 8, 2))
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
                        T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + (thread_binding // 64 * 2048 + ki * 1024 + thread_binding % 16 // 8 * 512 + (thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512), 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local, i * 8, 8, 2))
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
                    T.ptx_ldmatrix(T.bool(True), 4, T.tvm_access_ptr(T.type_annotation("float16"), buf_dyn_shmem, 12288 + (thread_binding // 64 * 2048 + ki * 1024 + thread_binding % 16 // 8 * 512 + (thread_binding % 16 * 64 + (thread_binding % 8 // 4 + i // 2) % 2 * 32 + (thread_binding % 4 // 2 + i % 2) % 2 * 16 + (thread_binding % 32 // 16 + thread_binding % 2) % 2 * 8) % 512 + 4096), 8, 1), T.tvm_access_ptr(T.type_annotation("float16"), B_local, i * 8, 8, 2))
                for i, j in T.grid(4, 4):
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i * 8, B_local, j * 8, C_local, i * 32 + j * 8, T.bool(False))
                    T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local, i * 8, B_local, j * 8 + 4, C_local, i * 32 + j * 8 + 4, T.bool(False))
        for i in T.unroll(64):
            C_local_cast_2 = T.Buffer((2,), "float16", data=C_local_cast, scope="local")
            C_local_3 = T.Buffer((128,), data=C_local, scope="local")
            C_local_cast_2[0:2] = T.Cast("float16x2", C_local_3[i * 2:i * 2 + 2])
            C_2 = T.Buffer((1048576,), "float16", data=C)
            C_2[by * 131072 + thread_binding % 64 // 32 * 65536 + i // 16 * 16384 + i % 2 * 8192 + thread_binding % 32 // 4 * 1024 + bx * 128 + thread_binding // 64 * 64 + i % 16 // 2 * 8 + thread_binding % 4 * 2:by * 131072 + thread_binding % 64 // 32 * 65536 + i // 16 * 16384 + i % 2 * 8192 + thread_binding % 32 // 4 * 1024 + bx * 128 + thread_binding // 64 * 64 + i % 16 // 2 * 8 + thread_binding % 4 * 2 + 2] = C_local_cast_2[0:2]