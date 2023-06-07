
import torch
import torch.nn.functional as F
from typing import Callable


def softmax(x, dim: int, onnx_trace: bool = False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


def get_activation_fn(activation: str) -> Callable:
    """Returns the activation function corresponding to `activation`"""
    # from fairseq.modules import gelu, gelu_accurate

    if activation == "relu":
        return F.relu
    # elif activation == "relu_squared":
    #     return relu_squared
    # elif activation == "gelu":
    #     return gelu
    # elif activation == "gelu_fast":
    #     deprecation_warning(
    #         "--activation-fn=gelu_fast has been renamed to gelu_accurate"
    #     )
    #     return gelu_accurate
    # elif activation == "gelu_accurate":
    #     return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))
