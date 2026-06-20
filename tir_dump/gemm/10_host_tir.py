# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def matmul(self_handle: T.handle, args: T.handle, num_args: T.int32, result: T.handle("void", "global")) -> T.int32:
        T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_matmul", "target": T.target({"keys": ["cpu"], "kind": "llvm", "mtriple": "", "tag": ""}), "thread_extent": {}, "tilelang_out_idx": [-1], "tir.is_entry_func": True, "tl.has_tma": T.bool(False), "tl.readonly_param_indices": [0, 1, 2], "tma_descriptor_args": {}})
        assert num_args == 3, "matmul: num_args should be 3"
        assert not T.isnullptr(args), "matmul: args pointer is NULL"
        A_type_index: T.int32 = T.tvm_struct_get(args, 0, 13, "int32")
        assert A_type_index == 0 or A_type_index == 4 or A_type_index == 7 or 64 <= A_type_index, "kernel matmul input A expected pointer or tensor handle"
        B_type_index: T.int32 = T.tvm_struct_get(args, 1, 13, "int32")
        assert B_type_index == 0 or B_type_index == 4 or B_type_index == 7 or 64 <= B_type_index, "kernel matmul input B expected pointer or tensor handle"
        C_handle_type_index: T.int32 = T.tvm_struct_get(args, 2, 13, "int32")
        assert C_handle_type_index == 0 or C_handle_type_index == 4 or C_handle_type_index == 7 or 64 <= C_handle_type_index, "kernel matmul input C expected pointer or tensor handle"
        A: T.handle = T.Select(A_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 0, 15, "handle"), 24), T.tvm_struct_get(args, 0, 15, "handle"))
        B: T.handle = T.Select(B_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 1, 15, "handle"), 24), T.tvm_struct_get(args, 1, 15, "handle"))
        C_handle: T.handle = T.Select(C_handle_type_index == 70, T.handle_add_byte_offset(T.tvm_struct_get(args, 2, 15, "handle"), 24), T.tvm_struct_get(args, 2, 15, "handle"))
        matmul_A_is_null: T.bool = T.isnullptr(A)
        assert not matmul_A_is_null, "matmul.A is expected to have non-NULL pointer"
        matmul_B_is_null: T.bool = T.isnullptr(B)
        assert not matmul_B_is_null, "matmul.B is expected to have non-NULL pointer"
        matmul_C_is_null: T.bool = T.isnullptr(C_handle)
        assert not matmul_C_is_null, "matmul.C is expected to have non-NULL pointer"
        matmul_A_shape: T.handle("int64", "global") = T.tvm_struct_get(A, 0, 2, "handle")
        matmul_A_shape_1 = T.decl_buffer((2,), "int64", data=matmul_A_shape)
        matmul_B_shape: T.handle("int64", "global") = T.tvm_struct_get(B, 0, 2, "handle")
        matmul_B_shape_1 = T.decl_buffer((2,), "int64", data=matmul_B_shape)
        matmul_C_shape: T.handle("int64", "global") = T.tvm_struct_get(C_handle, 0, 2, "handle")
        matmul_C_shape_1 = T.decl_buffer((2,), "int64", data=matmul_C_shape)
        if T.tvm_struct_get(A, 0, 4, "int32") != 2:
            T.call_packed("__tvm_error_ndim_mismatch", "matmul", "A", T.int64(2), T.Cast("int64", T.tvm_struct_get(A, 0, 4, "int32")))
        matmul_A_strides: T.handle("int64", "global") = T.tvm_struct_get(A, 0, 3, "handle")
        matmul_A_strides_1 = T.decl_buffer((2,), "int64", data=matmul_A_strides)
        dev_id: T.int32 = T.tvm_struct_get(A, 0, 9, "int32")
        A_1: T.handle("float16", "global") = T.tvm_struct_get(A, 0, 1, "handle")
        T.attr(A_1, "storage_alignment", 64)
        if T.tvm_struct_get(B, 0, 4, "int32") != 2:
            T.call_packed("__tvm_error_ndim_mismatch", "matmul", "B", T.int64(2), T.Cast("int64", T.tvm_struct_get(B, 0, 4, "int32")))
        matmul_B_strides: T.handle("int64", "global") = T.tvm_struct_get(B, 0, 3, "handle")
        matmul_B_strides_1 = T.decl_buffer((2,), "int64", data=matmul_B_strides)
        B_1: T.handle("float16", "global") = T.tvm_struct_get(B, 0, 1, "handle")
        T.attr(B_1, "storage_alignment", 64)
        if T.tvm_struct_get(C_handle, 0, 4, "int32") != 2:
            T.call_packed("__tvm_error_ndim_mismatch", "matmul", "C", T.int64(2), T.Cast("int64", T.tvm_struct_get(C_handle, 0, 4, "int32")))
        matmul_C_strides: T.handle("int64", "global") = T.tvm_struct_get(C_handle, 0, 3, "handle")
        matmul_C_strides_1 = T.decl_buffer((2,), "int64", data=matmul_C_strides)
        C: T.handle("float16", "global") = T.tvm_struct_get(C_handle, 0, 1, "handle")
        T.attr(C, "storage_alignment", 64)
        T.attr("default", "device_id", dev_id)
        T.attr("default", "device_type", 2)
        if T.tvm_struct_get(A, 0, 5, "uint8") != T.uint8(2) or T.tvm_struct_get(A, 0, 6, "uint8") != T.uint8(16) or T.tvm_struct_get(A, 0, 7, "uint16") != T.uint16(1):
            T.call_packed("__tvm_error_dtype_mismatch", "matmul", "A", T.Cast("int64", T.tvm_struct_get(A, 0, 5, "uint8")), T.Cast("int64", T.tvm_struct_get(A, 0, 6, "uint8")), T.Cast("int64", T.tvm_struct_get(A, 0, 7, "uint16")), T.int64(2), T.int64(16), T.int64(1))
        if T.Cast("int32", matmul_A_shape_1[0]) != 1024:
            T.call_packed("__tvm_error_expect_eq", "matmul", "A", "shape[0]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_A_shape_1[0])))
        if T.Cast("int32", matmul_A_shape_1[1]) != 1024:
            T.call_packed("__tvm_error_expect_eq", "matmul", "A", "shape[1]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_A_shape_1[1])))
        if T.if_then_else(T.isnullptr(matmul_A_strides), 1, T.Cast("int32", matmul_A_strides_1[1])) != 1:
            T.call_packed("__tvm_error_expect_eq", "matmul", "A", "strides[1]", T.int64(1), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_A_strides), 1, T.Cast("int32", matmul_A_strides_1[1]))))
        if T.if_then_else(T.isnullptr(matmul_A_strides), 1, T.Cast("int32", matmul_A_strides_1[0])) != 1024:
            T.call_packed("__tvm_error_expect_eq", "matmul", "A", "strides[0]", T.int64(1024), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_A_strides), 1, T.Cast("int32", matmul_A_strides_1[0]))))
        if T.uint64(0) != T.tvm_struct_get(A, 0, 8, "uint64"):
            T.call_packed("__tvm_error_byte_offset_mismatch", "matmul", "A", T.int64(0), T.Cast("int64", T.tvm_struct_get(A, 0, 8, "uint64")))
        if T.tvm_struct_get(A, 0, 10, "int32") != 2:
            T.call_packed("__tvm_error_device_type_mismatch", "matmul", "A", T.int64(2), T.Cast("int64", T.tvm_struct_get(A, 0, 10, "int32")))
        if T.isnullptr(A_1):
            T.call_packed("__tvm_error_null_ptr", "matmul", "A", "data pointer")
        if T.tvm_struct_get(B, 0, 5, "uint8") != T.uint8(2) or T.tvm_struct_get(B, 0, 6, "uint8") != T.uint8(16) or T.tvm_struct_get(B, 0, 7, "uint16") != T.uint16(1):
            T.call_packed("__tvm_error_dtype_mismatch", "matmul", "B", T.Cast("int64", T.tvm_struct_get(B, 0, 5, "uint8")), T.Cast("int64", T.tvm_struct_get(B, 0, 6, "uint8")), T.Cast("int64", T.tvm_struct_get(B, 0, 7, "uint16")), T.int64(2), T.int64(16), T.int64(1))
        if T.Cast("int32", matmul_B_shape_1[0]) != 1024:
            T.call_packed("__tvm_error_expect_eq", "matmul", "B", "shape[0]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_B_shape_1[0])))
        if T.Cast("int32", matmul_B_shape_1[1]) != 1024:
            T.call_packed("__tvm_error_expect_eq", "matmul", "B", "shape[1]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_B_shape_1[1])))
        if T.if_then_else(T.isnullptr(matmul_B_strides), 1, T.Cast("int32", matmul_B_strides_1[1])) != 1:
            T.call_packed("__tvm_error_expect_eq", "matmul", "B", "strides[1]", T.int64(1), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_B_strides), 1, T.Cast("int32", matmul_B_strides_1[1]))))
        if T.if_then_else(T.isnullptr(matmul_B_strides), 1, T.Cast("int32", matmul_B_strides_1[0])) != 1024:
            T.call_packed("__tvm_error_expect_eq", "matmul", "B", "strides[0]", T.int64(1024), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_B_strides), 1, T.Cast("int32", matmul_B_strides_1[0]))))
        if T.uint64(0) != T.tvm_struct_get(B, 0, 8, "uint64"):
            T.call_packed("__tvm_error_byte_offset_mismatch", "matmul", "B", T.int64(0), T.Cast("int64", T.tvm_struct_get(B, 0, 8, "uint64")))
        if T.tvm_struct_get(B, 0, 9, "int32") != T.tvm_struct_get(A, 0, 9, "int32"):
            T.call_packed("__tvm_error_expect_eq", "matmul", "B", "device_id", T.Cast("int64", T.tvm_struct_get(A, 0, 9, "int32")), T.Cast("int64", T.tvm_struct_get(B, 0, 9, "int32")))
        if T.tvm_struct_get(B, 0, 10, "int32") != 2:
            T.call_packed("__tvm_error_device_type_mismatch", "matmul", "B", T.int64(2), T.Cast("int64", T.tvm_struct_get(B, 0, 10, "int32")))
        if T.isnullptr(B_1):
            T.call_packed("__tvm_error_null_ptr", "matmul", "B", "data pointer")
        if T.tvm_struct_get(C_handle, 0, 5, "uint8") != T.uint8(2) or T.tvm_struct_get(C_handle, 0, 6, "uint8") != T.uint8(16) or T.tvm_struct_get(C_handle, 0, 7, "uint16") != T.uint16(1):
            T.call_packed("__tvm_error_dtype_mismatch", "matmul", "C", T.Cast("int64", T.tvm_struct_get(C_handle, 0, 5, "uint8")), T.Cast("int64", T.tvm_struct_get(C_handle, 0, 6, "uint8")), T.Cast("int64", T.tvm_struct_get(C_handle, 0, 7, "uint16")), T.int64(2), T.int64(16), T.int64(1))
        if T.Cast("int32", matmul_C_shape_1[0]) != 1024:
            T.call_packed("__tvm_error_expect_eq", "matmul", "C", "shape[0]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_C_shape_1[0])))
        if T.Cast("int32", matmul_C_shape_1[1]) != 1024:
            T.call_packed("__tvm_error_expect_eq", "matmul", "C", "shape[1]", T.int64(1024), T.Cast("int64", T.Cast("int32", matmul_C_shape_1[1])))
        if T.if_then_else(T.isnullptr(matmul_C_strides), 1, T.Cast("int32", matmul_C_strides_1[1])) != 1:
            T.call_packed("__tvm_error_expect_eq", "matmul", "C", "strides[1]", T.int64(1), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_C_strides), 1, T.Cast("int32", matmul_C_strides_1[1]))))
        if T.if_then_else(T.isnullptr(matmul_C_strides), 1, T.Cast("int32", matmul_C_strides_1[0])) != 1024:
            T.call_packed("__tvm_error_expect_eq", "matmul", "C", "strides[0]", T.int64(1024), T.Cast("int64", T.if_then_else(T.isnullptr(matmul_C_strides), 1, T.Cast("int32", matmul_C_strides_1[0]))))
        if T.uint64(0) != T.tvm_struct_get(C_handle, 0, 8, "uint64"):
            T.call_packed("__tvm_error_byte_offset_mismatch", "matmul", "C", T.int64(0), T.Cast("int64", T.tvm_struct_get(C_handle, 0, 8, "uint64")))
        if T.tvm_struct_get(C_handle, 0, 9, "int32") != T.tvm_struct_get(A, 0, 9, "int32"):
            T.call_packed("__tvm_error_expect_eq", "matmul", "C", "device_id", T.Cast("int64", T.tvm_struct_get(A, 0, 9, "int32")), T.Cast("int64", T.tvm_struct_get(C_handle, 0, 9, "int32")))
        if T.tvm_struct_get(C_handle, 0, 10, "int32") != 2:
            T.call_packed("__tvm_error_device_type_mismatch", "matmul", "C", T.int64(2), T.Cast("int64", T.tvm_struct_get(C_handle, 0, 10, "int32")))
        if T.isnullptr(C):
            T.call_packed("__tvm_error_null_ptr", "matmul", "C", "data pointer")
        A_2 = T.decl_buffer((1024, 1024), "float16", data=A_1, strides=(1024, 1))
        B_2 = T.decl_buffer((1024, 1024), "float16", data=B_1, strides=(1024, 1))
        C_1 = T.decl_buffer((1024, 1024), "float16", data=C, strides=(1024, 1))
        T.call_packed("__tvm_set_device", 2, dev_id)
        with T.attr(0, "compute_scope", "matmul_compute_"):
            T.call_packed("matmul_kernel", A_1, B_1, C, 8, 8, 128, 1, 1, 49152)
        return 0