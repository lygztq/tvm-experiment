import numpy as np
import tvm
from tvm import relay
from time import time

# actual input shape
batch_size = 128
input_dim = 1024
hidden_dim = 512
out_dim = 1024

# relay var shapes
v_batch_size = batch_size
v_input_dim = input_dim
v_out_dim = out_dim
v_hidden_dim = hidden_dim

# v_batch_size = relay.Any()
# v_input_dim = relay.Any()
# v_out_dim = relay.Any()
# v_hidden_dim = relay.Any()

# rt settings
exec_mod = "vm"
# tgt = "cuda -libs=cublas"
# tgt = "cuda"
tgt = "llvm"
dev = tvm.device(tgt)
act_run = True

# utils funcs
def dense_with_bias(x, w, b):
    o = relay.nn.dense(x, w)
    o = relay.nn.bias_add(o, b)
    return o

def TwoLayerFC():
    # ======= inputs ======= #
    x = relay.var("x", shape=(v_batch_size, v_input_dim))

    # ======= weights ====== #
    w_1 = relay.var("w_1", shape=(v_hidden_dim, v_input_dim))
    b_1 = relay.var("b_1", shape=(v_hidden_dim, ))
    w_2 = relay.var("w_2", shape=(v_out_dim, v_hidden_dim))
    b_2 = relay.var("b_2", shape=(v_out_dim, ))

    # ==== first layer ===== #
    a_1 = dense_with_bias(x, w_1, b_1)
    a_1 = relay.nn.relu(a_1)

    # ==== second layer ==== #
    a_2 = dense_with_bias(a_1, w_2, b_2)
    a_2 = relay.nn.softmax(a_2)

    mod = tvm.IRModule()
    mod["main"] = relay.Function(
        [x, w_1, b_1, w_2, b_2],
        a_2)
    return mod


mod = TwoLayerFC()
print("Raw module: \n{}".format(str(mod)))

opt_mod, _ = relay.optimize(mod, target=tgt)
print("Opt module: \n{}".format(str(opt_mod)))

if act_run:
    args = [
        np.random.randn(batch_size, input_dim).astype("float32"), # x
        np.random.randn(hidden_dim, input_dim).astype("float32"), # w_1
        np.random.randn(hidden_dim).astype("float32"),            # b_1
        np.random.randn(out_dim, hidden_dim).astype("float32"),   # w_2
        np.random.randn(out_dim).astype("float32")                # b_2
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
    # print(result)
