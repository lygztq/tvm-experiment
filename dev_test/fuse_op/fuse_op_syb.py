from typing import List
import numpy as np
import tvm
import math
from tvm import relay
from tvm.relay.prelude import Prelude
from tvm.relay import transform
from time import time

# ======== actual input shape ======== #
batch_size = 16
seq_len = 128
input_dim = 768
inner_dim = 4 * input_dim
num_heads = 12

# ======== var shapes ======== #
## config-1: only batch_size is dynamic
# v_batch_size = relay.Any()
# v_input_dim = input_dim
# v_inner_dim = inner_dim
# v_seq_len = seq_len
# v_num_heads = num_heads

# config-2: both seq_len and batch_size are dynamic
v_batch_size = tvm.te.var("batch_size")
v_input_dim = input_dim
v_inner_dim = inner_dim
v_seq_len = relay.Any()
v_num_heads = num_heads

# ======== rt settings ======== #
exec_mode = "vm"
# exec_mode = "debug"
# tgt_list = ["llvm", "cuda -libs=cublas,cudnn"]
# tgt_list = ["cuda -libs=cublas,cudnn"]
# tgt_list = ["llvm"]
# tgt_list = [(tgt, tvm.device(tgt)) for tgt in tgt_list]

# tgt = "llvm"
tgt = "cuda -libs=cublas"
dev = tvm.device(tgt)
act_run = True

def run_opt_pass(expr, opt_pass, import_prelude=False):
    assert isinstance(opt_pass, tvm.transform.Pass)
    mod = tvm.IRModule.from_expr(expr)
    if import_prelude:
        Prelude(mod)
    mod = relay.transform.InferType()(mod)
    mod = opt_pass(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def index_expr(idxs:List):
    return relay.const(np.array(idxs, dtype=np.int32), "int32")

def dyn_reshape(input_tensor, newshape:List):
    relay_newshape = relay.const(np.array(newshape, dtype=np.int32), dtype="int32")
    return relay.reshape(input_tensor, relay_newshape)

def syb_reshape(input_tensor, newshape):
    int_shape = []
    var_shape = []
    use_int = []
    foo_var = relay.const(0, "int32")
    for s in newshape:
        if isinstance(s, int):
            use_int.append(True)
            int_shape.append(s)
            var_shape.append(foo_var)
        else:
            use_int.append(False)
            int_shape.append(0)
            var_shape.append(relay.const(np.array(s, dtype=np.int32), "int32"))
    shape_var = relay.where(
        relay.const(np.array(use_int, dtype=np.int32), "bool"),
        relay.const(np.array(int_shape, dtype=np.int32), "int32"),
        relay.concatenate(var_shape, axis=0)
    )
    return relay.reshape(input_tensor, shape_var)

def gelu(x):
    cdf = relay.const(0.5, "float32") * (relay.const(1.0, "float32") + relay.tanh(
        relay.const(np.sqrt(2 / np.pi), "float32") * (x + relay.const(0.044715, "float32") * relay.power(x, relay.const(3, "float32")))
    ))
    return x * cdf

def BertForwardLayer():
    attention_mask = relay.var("input", shape=(v_batch_size, v_seq_len, v_seq_len), dtype="int")
    input_tensor = relay.var("input", shape=(v_batch_size, v_seq_len, v_input_dim))

    # =============== input parameters ======================== #
    w_k = relay.var("w_k", shape=(v_input_dim, v_input_dim))
    w_q = relay.var("w_q", shape=(v_input_dim, v_input_dim))
    w_v = relay.var("w_v", shape=(v_input_dim, v_input_dim))

    b_k = relay.var("b_k", shape=(v_input_dim,))
    b_q = relay.var("b_q", shape=(v_input_dim,))
    b_v = relay.var("b_v", shape=(v_input_dim,))

    # =============== get K, Q, and V ====================== #
    newshape = relay.take(relay.shape_of(input_tensor), index_expr([0, 1])) # (batch_size, seq_len)
    newshape = relay.concatenate([newshape, index_expr([v_num_heads, v_input_dim // v_num_heads])], axis=0) # (batch_size, seq_len, num_heads, input_dim // num_heads)
    input_tensor_2d = syb_reshape(input_tensor, [-3, v_input_dim]) # shape = (batch_size * seq_len, input_dim)

    k_tensor = relay.nn.dense(input_tensor_2d, w_k, units=v_input_dim) # shape = (batch_size * seq_len, input_dim)
    k_tensor = relay.nn.bias_add(k_tensor, b_k, axis=1) # shape = (batch_size * seq_len, input_dim)
    k_tensor = relay.reshape(k_tensor, newshape) # shape = (batch_size, seq_len, num_heads, input_dim // num_heads)
    k_tensor = relay.transpose(k_tensor, [0, 2, 1, 3]) # shape = (batch_size, num_heads, seq_len, input_dim // num_heads)

    q_tensor = relay.nn.dense(input_tensor_2d, w_q, units=v_input_dim)
    q_tensor = relay.nn.bias_add(q_tensor, b_q, axis=1)
    q_tensor = relay.reshape(q_tensor, newshape)
    q_tensor = relay.transpose(q_tensor, [0, 2, 1, 3]) # shape = (batch_size, num_heads, seq_len, input_dim // num_heads)

    v_tensor = relay.nn.dense(input_tensor_2d, w_v, units=v_input_dim)
    v_tensor = relay.nn.bias_add(v_tensor, b_v, axis=1)
    v_tensor = relay.reshape(v_tensor, newshape)
    v_tensor = relay.transpose(v_tensor, [0, 2, 1, 3]) # shape = (batch_size, num_heads, seq_len, input_dim // num_heads)

    # =============== attention =========================== #
    k_tensor = syb_reshape(k_tensor, [-3, -1, v_input_dim // v_num_heads])
    q_tensor = syb_reshape(q_tensor, [-3, -1, v_input_dim // v_num_heads])
    # q_tensor = relay.transpose(q_tensor, [0, 2, 1])
    attention_scores = relay.nn.batch_matmul(k_tensor, q_tensor) # shape = (batch_size * num_heads, seq_len, seq_len)
    attention_scores_newshape = relay.gather(relay.shape_of(v_tensor), axis=0, indices=relay.const(np.array([0, 1, 2, 2], dtype=np.int32), dtype="int32"))
    attention_scores = relay.reshape(attention_scores, attention_scores_newshape) # shape = (batch_size, num_heads, seq_len, seq_len)
    attention_scores = relay.multiply(attention_scores, relay.const(1.0 / math.sqrt(v_input_dim / v_num_heads), dtype="float32"))
    adder = relay.multiply(
                relay.subtract(
                    relay.const(1.0, dtype="float32"),
                    relay.cast(relay.expand_dims(attention_mask, axis=1), dtype="float32")),
                relay.const(-10000.0, dtype="float32"))
    attention_scores = relay.add(attention_scores, adder)
    attention_probs = relay.nn.softmax(attention_scores) # shape = (batch_size, num_heads, seq_len, seq_len)
    lastshape = relay.concatenate([
        relay.take(relay.shape_of(attention_probs), index_expr([0, 2])),
        relay.const(np.array([v_input_dim]), "int32")
    ], 0)
    attention_probs = relay.reshape(attention_probs, [-3, 0, 0]) # shape = (batch_size * num_heads, seq_len, seq_len)

    # =============== attention mask ======================= #
    v_tensor = relay.reshape(v_tensor, [-3, 0, 0]) # shape = (batch_size * num_heads, seq_len, input_dim // num_heads)
    v_tensor = relay.transpose(v_tensor, [0, 2, 1])
    attention_out = relay.nn.batch_matmul(attention_probs, v_tensor) # shape = (batch_size * num_heads, seq_len, input_dim // num_heads)
    attention_out = relay.reshape(attention_out, lastshape) # shape = (batch_size, seq_len, input_dim)

    # =============== out 1 ======================== #
    w_attr_out = relay.var("w_attr_out", shape=(v_input_dim, v_input_dim))
    b_attr_out = relay.var("b_attr_out", shape=(v_input_dim, ))

    attention_out = relay.reshape(attention_out, [-3, 0]) # shape = (batch_size * seq_len, input_dim)
    out_1 = relay.nn.dense(attention_out, w_attr_out) # shape = (batch_size * seq_len, input_dim)
    out_1 = relay.nn.bias_add(out_1, b_attr_out) # shape = (batch_size * seq_len, input_dim)
    out_1 = relay.add(out_1, input_tensor_2d) # shape = (batch_size * seq_len, input_dim)
    # out_1 = relay.nn.layer_norm(out_1, gamma=relay.ones((input_dim, ), "float32"), beta=relay.zeros((input_dim, ), "float32"), axis=-1)

    # =============== out 2 ======================== #
    w_inter = relay.var("w_inter", shape=(v_inner_dim, v_input_dim))
    b_inter = relay.var("b_inter", shape=(v_inner_dim, ))

    out_2 = relay.nn.dense(out_1, w_inter)
    out_2 = relay.nn.bias_add(out_2, b_inter)
    out_2 = gelu(out_2)

    # =============== out 3 ======================== #
    w_out = relay.var("w_out", shape=(v_input_dim, v_inner_dim))
    b_out = relay.var("b_out", shape=(v_input_dim, ))

    out_3 = relay.nn.dense(out_2, w_out)
    out_3 = relay.nn.bias_add(out_3, b_out)
    out_3 = relay.add(out_3, out_1)
    # out_3 = relay.nn.layer_norm(out_3, gamma=relay.ones((input_dim, ), "float32"), beta=relay.zeros((input_dim, ), "float32"), axis=-1)

    mod = tvm.IRModule()
    mod["main"] = relay.Function([
        w_k, w_q, w_v,
        b_k, b_q, b_v,
        w_attr_out, b_attr_out,
        w_inter, b_inter,
        w_out, b_out,
        input_tensor, attention_mask], out_3)
    return mod

mod = BertForwardLayer()
# print(str(mod))

# fuse_mod = run_opt_pass(mod["main"], transform.FuseOps())
# print(str(fuse_mod))
# opt_mod, _ = relay.optimize(mod, target=tgt)
# print(str(opt_mod))

if act_run:
    args = [
        np.random.randn(input_dim, input_dim).astype("float32"), # w_k
        np.random.randn(input_dim, input_dim).astype("float32"), # w_q
        np.random.randn(input_dim, input_dim).astype("float32"), # w_v
        np.random.randn(input_dim, ).astype("float32"), # b_k
        np.random.randn(input_dim, ).astype("float32"), # b_q
        np.random.randn(input_dim, ).astype("float32"), # b_v
        np.random.randn(input_dim, input_dim).astype("float32"), # w_attr_out
        np.random.randn(input_dim, ).astype("float32"), # b_attr_out
        np.random.randn(inner_dim, input_dim).astype("float32"), # w_inter
        np.random.randn(inner_dim, ).astype("float32"), # b_inter
        np.random.randn(input_dim, inner_dim).astype("float32"), # w_out
        np.random.randn(input_dim, ).astype("float32"), # b_out
        np.random.uniform(size=(batch_size, seq_len, input_dim)).astype("float32"), # input_tensor
        np.random.randint(2, size=(batch_size, seq_len, seq_len), dtype=np.int32) # attention_mask
    ]

    print("Running on ({}, {})".format(tgt, dev))
    if exec_mode == "debug" and dev.device_type != tvm.cpu().device_type:
        print("Pass, device!=cpu but exec_mod=debug")
        exit(0)
    ex = relay.create_executor(exec_mode, mod=mod, device=dev, target=tgt)
    
    # warm up
    _ = ex.evaluate()(*args)

    s_time = time()
    result = ex.evaluate()(*args)
    print("Time: {:.5f} ms".format(1000 * (time() - s_time)))
