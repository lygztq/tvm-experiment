import tvm
import tvm.testing
from tvm import te
import numpy as np
from tvm.contrib import tedd

# You will get better performance if you can identify the CPU you are targeting
# and specify it. If you're using llvm, you can get this information from the
# command ``llc --version`` to get the CPU type, and you can check
# ``/proc/cpuinfo`` for additional extensions that your processor might
# support. For example, you can use "llvm -mcpu=skylake-avx512" for CPUs with
# AVX-512 instructions.

tgt = tvm.target.Target(target="llvm", host="llvm")

# Recreate the schedule, since we modified it with the parallel operation in
# the previous example
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

s = te.create_schedule(C.op)

# This factor should be chosen to match the number of threads appropriate for
# your CPU. This will vary depending on architecture, but a good rule is
# setting this factor to equal the number of available CPU cores.
factor = 4

outer, inner = s[C].split(C.op.axis[0], factor=factor)
s[C].parallel(outer)
s[C].vectorize(inner)

tedd.viz_dataflow_graph(s, dot_file_path="./dev_test/flow.dot")
s = s.normalize()
tedd.viz_schedule_tree(s, dot_file_path="./dev_test/tree.dot")

fadd_vector = tvm.build(s, [A, B, C], tgt, name="myadd_parallel")

log = []
def evaluate_addition(func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    n = 32768
    a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
    b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), dev)
    c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)

    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = evaluator(a, b, c).mean
    print("%s: %f" % (optimization, mean_time))

    log.append((optimization, mean_time))

evaluate_addition(fadd_vector, tgt, "vector", log=log)

print(tvm.lower(s, [A, B, C], simple_mode=True))
