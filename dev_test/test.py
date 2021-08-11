import tvm
from tvm import relay
import numpy as np

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

# x = [[1, 2], [3, 4]]
# y = [[5, 6], [7, 8]]
# condition = [[0, 1], [-1, 0]]
# relay.where(condition, x, y)
