import numpy as np
import tvm
from tvm import relay
from time import time

from tvm.relay.backend.vm import VMCompiler

# actual input shape
# dim0 = 2
# dim1 = 10
# dim2 = 8
# dim3 = 16

dim0 = 2
dim1 = 10
dim2 = 257
dim3 = 1025

# relay var shapes
v_dim0 = relay.Any()
v_dim1 = relay.Any()
v_dim2 = relay.Any()
v_dim3 = relay.Any()

# static
v_dim0 = dim0
v_dim1 = dim1
v_dim2 = dim2
v_dim3 = dim3

# symbolic
# v_dim0 = tvm.te.var("d0")
# v_dim1 = tvm.te.var("d1")
# v_dim2 = tvm.te.var("d2")
# v_dim3 = tvm.te.var("d3")

# v_dim0 = tvm.tir.SizeVar("d0", dtype="int32")
# v_dim1 = tvm.tir.SizeVar("d1", dtype="int32")
# v_dim2 = tvm.tir.SizeVar("d2", dtype="int32")
# v_dim3 = tvm.tir.SizeVar("d3", dtype="int32")

# rt settings
exec_mod = "vm"
# tgt = "cuda -libs=cublas"
tgt = "cuda"
# tgt = "llvm"
dev = tvm.device(tgt)
act_run = True

def get_mod():
    x = relay.var("x", shape=(v_dim0, v_dim1, v_dim2, v_dim3))
    y = relay.nn.softmax(x)
    mod = tvm.IRModule()
    mod["main"] = relay.Function([x], y)
    return mod

mod = get_mod()
print("Raw module: \n{}".format(str(mod)))

# complier = VMCompiler()
# complier.lower(mod, tgt)
# l_mod = relay.vm.compile(mod, target=tgt)
# print(l_mod)

if act_run:
    args = [
        np.random.randn(dim0, dim1, dim2, dim3).astype("float32"), # x
    ]

    print("Running on ({}, {})".format(tgt, dev))
    if exec_mod == "debug" and dev.device_type != tvm.cpu().device_type:
        print("only cpu on debug exec mod, pass")
        exit(0)
    ex = relay.create_executor(exec_mod, mod=mod, device=dev, target=tgt)

    # warmup
    _ = ex.evaluate()(*args)
    
    s_time = time()
    result = ex.evaluate()(*args)
    print("Finish in {:.5f} ms".format(1000 * (time() - s_time)))
