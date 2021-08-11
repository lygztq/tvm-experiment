import numpy as np
import tvm
from tvm import relay

exec_mode = "vm"
# exec_mode = "debug"
# tgt_list = ["llvm", "cuda -libs=cublas,cudnn"]
# tgt_list = ["cuda -libs=cublas,cudnn"]
tgt_list = ["cuda"]
tgt_list = [(tgt, tvm.device(tgt)) for tgt in tgt_list]

# a = relay.var("a", shape=(relay.Any(), 4))
# b = relay.var("b", shape=(relay.Any(), 4))

a = relay.var("a", shape=(3, 4))
b = relay.var("b", shape=(3, 4))

c = relay.add(a, b)
# d = relay.reshape(c, [-3])
# e = relay.shape_of(d)
# f = relay.take(e, relay.const(0, "int32"))
# g = relay.cast(f, "int32")
# h = relay.arange(g, dtype="int32")
# i = relay.shape_of(h)
# out = i

e = relay.nn.softmax(c)
f = relay.add(e, b)
out = f

mod = tvm.IRModule()
mod["main"] = relay.Function([a, b], out)

print(mod)

args = [
    np.random.randn(3, 4).astype(np.float32),
    np.random.randn(3, 4).astype(np.float32)
]

mod, _ = relay.optimize(mod, target=tgt_list[0][0])
print(mod)
ex = relay.create_executor(exec_mode, mod=mod, target=tgt_list[0][0])
code = ex.executable.bytecode
print(code)
print(ex.executable.stats)
result = ex.evaluate()(*args)
print(result)