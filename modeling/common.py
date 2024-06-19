from typing import Literal, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hswish(nn.Module):
    def __init__(self, inplace: bool = True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


# out = max(0, min(1, slop*x+offset))
# paddle.fluid.layers.hard_sigmoid(x, slope=0.2, offset=0.5, name=None)
class Hsigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torch: F.relu6(x + 3., inplace=self.inplace) / 6.
        # paddle: F.relu6(1.2 * x + 3., inplace=self.inplace) / 6.
        return F.relu6(1.2 * x + 3.0, inplace=self.inplace) / 6.0


class GELU(nn.Module):
    def __init__(self, inplace: bool = True):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)


class Swish(nn.Module):
    def __init__(self, inplace: bool = True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


ActivationOptions: TypeAlias = Literal[
    "relu",
    "relu6",
    "sigmoid",
    "hard_sigmoid",
    "hard_swish",
    "leakyrelu",
    "gelu",
    "swish",
]


class Activation(nn.Module):
    def __init__(self, act_type: ActivationOptions, inplace: bool = True):
        super(Activation, self).__init__()
        match act_type:
            case "relu":
                self.act = nn.ReLU(inplace=inplace)
            case "relu6":
                self.act = nn.ReLU6(inplace=inplace)
            case "sigmoid":
                raise NotImplementedError
            case "hard_sigmoid":
                self.act = Hsigmoid(inplace)  # nn.Hardsigmoid(inplace=inplace)#Hsigmoid(inplace)#
            case "hard_swish":
                self.act = Hswish(inplace=inplace)
            case "leakyrelu":
                self.act = nn.LeakyReLU(inplace=inplace)
            case "gelu":
                self.act = GELU(inplace=inplace)
            case "swish":
                self.act = Swish(inplace=inplace)
            case _:
                raise NotImplementedError

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.act(inputs)
