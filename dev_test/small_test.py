import tvm
from tvm import te

n = tvm.te.var("n")
m = tvm.te.var("m")
# k = tvm.te.var("k")
k = tvm.tir.Any()

A = te.placeholder((m, n), name="A")
B = te.placeholder((n, k), name="B")

vec = 32
# k = te.reduce_axis((0, k // vec), "k")
k = te.reduce_axis((0, tvm.tir.FloorDiv(k, vec)), "k")
CC = te.compute((m, n, vec), lambda z, y, x: te.sum(A[z, k * vec + x] * B[y, k * vec + x], axis=k))
kk = te.reduce_axis((0, vec), "kk")
C = te.compute((m, n), lambda y, x: te.sum(CC[y, x, kk], axis=kk), tag="dense_nopack")

print(k // vec)

