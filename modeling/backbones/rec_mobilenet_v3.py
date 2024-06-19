import os
import sys
from typing import Annotated, List, Literal

import torch
import torch.nn as nn

from .det_mobilenet_v3 import ConvBNLayer, ResidualUnit, make_divisible


class MobileNetV3(nn.Module):
    small_strides = [2, 2, 2, 2]
    large_strides = [1, 2, 2, 2]

    def __init__(
        self,
        in_channels: int = 3,
        model_name: Literal["large", "small"] = "small",
        scale: Literal["0.35", "0.5", "0.75", "1.0", "1.25"] = "0.5",
        strides: Annotated[List[int], 4] | None = None,
        **kwargs,
    ):
        super(MobileNetV3, self).__init__()

        if strides is None:
            match model_name:
                case "large":
                    strides = self.large_stride

                case "small":
                    strides = self.small_stride

        match model_name:
            case "large":
                large_strides = strides or self.large_strides
                cfg = [
                    # k, exp, c,  se,     nl,  s,
                    [3, 16, 16, False, "relu", large_strides[0]],
                    [3, 64, 24, False, "relu", (large_strides[1], 1)],
                    [3, 72, 24, False, "relu", 1],
                    [5, 72, 40, True, "relu", (large_strides[2], 1)],
                    [5, 120, 40, True, "relu", 1],
                    [5, 120, 40, True, "relu", 1],
                    [3, 240, 80, False, "hard_swish", 1],
                    [3, 200, 80, False, "hard_swish", 1],
                    [3, 184, 80, False, "hard_swish", 1],
                    [3, 184, 80, False, "hard_swish", 1],
                    [3, 480, 112, True, "hard_swish", 1],
                    [3, 672, 112, True, "hard_swish", 1],
                    [5, 672, 160, True, "hard_swish", (large_strides[3], 1)],
                    [5, 960, 160, True, "hard_swish", 1],
                    [5, 960, 160, True, "hard_swish", 1],
                ]
                cls_ch_squeeze = 960
            case "small":
                small_strides = strides or self.small_strides
                cfg = [
                    # k, exp, c,  se,     nl,  s,
                    [3, 16, 16, True, "relu", (small_strides[0], 1)],
                    [3, 72, 24, False, "relu", (small_strides[1], 1)],
                    [3, 88, 24, False, "relu", 1],
                    [5, 96, 40, True, "hard_swish", (small_strides[2], 1)],
                    [5, 240, 40, True, "hard_swish", 1],
                    [5, 240, 40, True, "hard_swish", 1],
                    [5, 120, 48, True, "hard_swish", 1],
                    [5, 144, 48, True, "hard_swish", 1],
                    [5, 288, 96, True, "hard_swish", (small_strides[3], 1)],
                    [5, 576, 96, True, "hard_swish", 1],
                    [5, 576, 96, True, "hard_swish", 1],
                ]
                cls_ch_squeeze = 576

        inplanes = 16
        scale = float(scale)
        # conv1
        self.conv1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act="hard_swish",
            name="conv1",
        )

        i = 0
        block_list = []
        inplanes = make_divisible(inplanes * scale)
        for k, exp, c, se, nl, s in cfg:
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name="conv" + str(i + 2),
                )
            )
            inplanes = make_divisible(scale * c)
            i += 1
        self.blocks = nn.Sequential(*block_list)

        self.conv2 = ConvBNLayer(
            in_channels=inplanes,
            out_channels=make_divisible(scale * cls_ch_squeeze),
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            if_act=True,
            act="hard_swish",
            name="conv_last",
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = make_divisible(scale * cls_ch_squeeze)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x
