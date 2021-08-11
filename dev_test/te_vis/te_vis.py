import tvm
from tvm import te
from tvm.contrib import tedd

import os

curr_path = os.path.dirname(os.path.abspath(__file__))

n = te.var("n", "int32")
m = te.var("m", "int32")
a = te.placeholder((n, ), name="a")
b = te.placeholder((n, ), name="b")
c = te.compute((n, ), lambda i: a[i] + b[i], name="c")

s = te.create_schedule([c.op])
# tedd.viz_itervar_relationship_graph(s, dot_file_path=os.path.join(curr_path, "normal.dot"))

# # xo, xi = s[c].split(c.op.axis[0], factor=32)
# # # tedd.viz_itervar_relationship_graph(s, dot_file_path=os.path.join(curr_path, "split.dot"))
# # xio, xii = s[c].split(xi, factor=32)
# # tedd.viz_itervar_relationship_graph(s, dot_file_path=os.path.join(curr_path, "split_split.dot"))

# bx, tx = s[c].split(c.op.axis[0], factor=32)
# s[c].bind(bx, te.thread_axis("blockIdx.x"))
# s[c].bind(tx, te.thread_axis("threadIdx.x"))
# tedd.viz_itervar_relationship_graph(s, dot_file_path=os.path.join(curr_path, "split-bind.dot"))

# A = te.placeholder((m, n), name="A")
# B = te.compute((m, n), lambda i, j: A[i, j], name="B")

# s = te.create_schedule(B.op)
# xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# tedd.viz_itervar_relationship_graph(s, dot_file_path=os.path.join(curr_path, "tile.dot"))

# A = te.placeholder((m, n), name="A")
# B = te.compute((m, n), lambda i, j: A[i, j], name="B")

# s = te.create_schedule(B.op)
# # tile to four axes first: (i.outer, j.outer, i.inner, j.inner)
# xo, yo, xi, yi = s[B].tile(B.op.axis[0], B.op.axis[1], x_factor=10, y_factor=5)
# # then reorder the axes: (i.inner, j.outer, i.outer, j.inner)
# s[B].reorder(xi, yo, xo, yi)
# tedd.viz_itervar_relationship_graph(s, dot_file_path=os.path.join(curr_path, "reorder.dot"))

