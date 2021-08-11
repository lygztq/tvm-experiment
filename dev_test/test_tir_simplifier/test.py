import tvm

ib = tvm.tir.ir_builder.create()
a = tvm.tir.SizeVar("a", "int32")
b = tvm.tir.SizeVar("b", "int32")
c = tvm.tir.SizeVar("c", "int32")

k = tvm.tir.SizeVar("k", "int32")
n = tvm.tir.SizeVar("n", "int32")

A = ib.allocate("float32", n, name="A")
abuffer = tvm.tir.decl_buffer((n, ), dtype=A.dtype, data=A.asobject())
with ib.for_range(0, k, name="i") as i:
    with ib.if_scope(ib.likely(i < (a + (b + c)))):
        with ib.if_scope(ib.likely(i < (c + (b + a)))):
            A[i] = A[i] + 1

stmt = ib.get()

mod = tvm.IRModule.from_expr(tvm.tir.PrimFunc([a, b, c, k, n, A], stmt))
print("Before: ", mod)
mod_after = tvm.tir.transform.Simplify()(mod)
print("After: ", mod_after)
