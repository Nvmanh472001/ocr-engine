import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.backbones.det_mobilenet_v3 import SEModule

from .intracl import IntraCLBlock


def hard_swish(x: torch.Tensor, inplace=True) -> torch.Tensor:
    return x * F.relu6(x + 3.0, inplace=inplace) / 6.0


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int | Tuple[int, int],
        stride: int = 1,
        groups: int = None,
        if_act: bool = True,
        act: str = "relu",
        **kwargs,
    ):
        super(DSConv, self).__init__()
        if groups == None:
            groups = in_channels
        self.if_act = if_act
        self.act = act
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=int(in_channels * 4),
            kernel_size=1,
            stride=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(int(in_channels * 4))

        self.conv3 = nn.Conv2d(
            in_channels=int(in_channels * 4),
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self._c = [in_channels, out_channels]
        if in_channels != out_channels:
            self.conv_end = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                bias=False,
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.conv1(inputs)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = hard_swish(x)
            else:
                print("The activation function({}) is selected incorrectly.".format(self.act))
                exit()

        x = self.conv3(x)
        if self._c[0] != self._c[1]:
            x = x + self.conv_end(inputs)
        return x


class DBFPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_asf: bool = False, **kwargs):
        super(DBFPN, self).__init__()
        self.out_channels = out_channels
        self.use_asf = use_asf

        self.in2_conv = nn.Conv2d(in_channels=in_channels[0], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(in_channels=in_channels[1], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in4_conv = nn.Conv2d(in_channels=in_channels[2], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(in_channels=in_channels[3], out_channels=self.out_channels, kernel_size=1, bias=False)
        self.p5_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.p4_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.p3_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.p2_conv = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        if self.use_asf is True:
            self.asf = ASFBlock(self.out_channels, self.out_channels // 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c2, c3, c4, c5 = x

        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")  # align_mode=1)  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # align_mode=1)  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # align_mode=1)  # 1/4

        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")  # align_mode=1)
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")  # align_mode=1)
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")  # align_mode=1)

        fuse = torch.cat([p5, p4, p3, p2], dim=1)

        if self.use_asf is True:
            fuse = self.asf(fuse, [p5, p4, p3, p2])

        return fuse


class RSELayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, shortcut: bool = True):
        super(RSELayer, self).__init__()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            padding=int(kernel_size // 2),
            bias=False,
        )
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def forward(self, ins: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(ins)
        if self.shortcut:
            out = x + self.se_block(x)
        else:
            out = self.se_block(x)
        return out


class RSEFPN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True, **kwargs):
        super(RSEFPN, self).__init__()
        self.out_channels = out_channels
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        self.intracl = False
        if "intracl" in kwargs.keys() and kwargs["intracl"] is True:
            self.intracl = kwargs["intracl"]
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

        for i in range(len(in_channels)):
            self.ins_conv.append(RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=shortcut))
            self.inp_conv.append(RSELayer(out_channels, out_channels // 4, kernel_size=3, shortcut=shortcut))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        out4 = in4 + F.interpolate(in5, scale_factor=2, mode="nearest")  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2, mode="nearest")  # 1/4

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = F.interpolate(p5, scale_factor=8, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2, mode="nearest")

        fuse = torch.cat([p5, p4, p3, p2], dim=1)
        return fuse


class ASFBlock(nn.Module):
    """
    This code is refered from:
        https://github.com/MhLiao/DB/blob/master/decoders/feature_attention.py
    """

    def __init__(self, in_channels: int, inter_channels: int, out_features_num: int = 4):
        """
        Adaptive Scale Fusion (ASF) block of DBNet++
        Args:
            in_channels: the number of channels in the input data
            inter_channels: the number of middle channels
            out_features_num: the number of fused stages
        """
        super(ASFBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_features_num = out_features_num
        self.conv = nn.Conv2d(in_channels, inter_channels, 3, padding=1)

        self.spatial_scale = nn.Sequential(
            # Nx1xHxW
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        self.channel_scale = nn.Sequential(
            nn.Conv2d(in_channels=inter_channels, out_channels=out_features_num, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, fuse_features: torch.Tensor, features_list: list[torch.Tensor]) -> torch.Tensor:
        fuse_features = self.conv(fuse_features)
        spatial_x = torch.mean(fuse_features, dim=1, keepdim=True)
        attention_scores = self.spatial_scale(spatial_x) + fuse_features
        attention_scores = self.channel_scale(attention_scores)
        assert len(features_list) == self.out_features_num

        out_list = []
        for i in range(self.out_features_num):
            out_list.append(attention_scores[:, i : i + 1] * features_list[i])
        return torch.cat(out_list, dim=1)
