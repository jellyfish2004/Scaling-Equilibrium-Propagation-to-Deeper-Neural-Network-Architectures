import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F


class Interaction(nn.Module):
    def __init__(self):
        super().__init__()

    def create_state(self, batch_size, device):
        raise NotImplementedError

    def energy(self, pre, post):
        raise NotImplementedError

    def backward(self, pre, post):
        raise NotImplementedError


class InteractionConv2d(Interaction):
    def __init__(self, in_ch, out_ch, kernel_size, h_out, w_out,
                 stride=1, padding=0, bias=True, weight_gain=None, bias_gain=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch, 1, 1))
            self._bias_is_param = True
        else:
            self.register_buffer("bias", torch.zeros(out_ch, 1, 1))
            self._bias_is_param = False

        self.h_out = h_out
        self.w_out = w_out

        # weight init
        if weight_gain is not None:
            kH = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            kW = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
            size_pre = in_ch * kH * kW
            scale = weight_gain * (1.0 / size_pre) ** 0.5
            nn.init.uniform_(self.conv.weight, -scale, +scale)
        if bias and bias_gain is not None:
            nn.init.uniform_(self.bias, -bias_gain, +bias_gain)

        # Pre-compute int tuple stride for output_padding computation
        s = self.conv.stride
        if isinstance(s, int):
            self._stride_tuple = (s, s)
        else:
            self._stride_tuple = tuple(s)

    def create_state(self, batch_size, device):
        # allocate in channels_last directly
        state = torch.zeros(batch_size, self.conv.out_channels, self.h_out, self.w_out,
                            device=device, requires_grad=False).contiguous(memory_format=torch.channels_last)
        return state

    def energy(self, pre, post):
        feat = self.conv(pre)
        e = -(feat * post).sum(dim=(1, 2, 3))
        bias = self.bias.view(1, -1, 1, 1)
        e += -(bias * post).sum(dim=(1, 2, 3))
        return e

    def backward(self, pre, post):
        with torch.no_grad():
            feat = self.conv(pre)
            bias = self.bias.view(1, -1, 1, 1)
            grad_post = -(feat + bias).contiguous(memory_format=torch.channels_last)

            output_padding = (self._stride_tuple[0] - 1, self._stride_tuple[1] - 1)

            grad_pre = -F.conv_transpose2d(
                input=post,
                weight=self.conv.weight,
                stride=self.conv.stride,
                padding=self.conv.padding,
                output_padding=output_padding
            ).contiguous(memory_format=torch.channels_last)

        return grad_post, grad_pre


class InteractionConvMaxPool2d(Interaction):
    def __init__(self, in_ch, out_ch, kernel_size, h_out, w_out,
                 stride=1, padding=0, bias=True, weight_gain=None, bias_gain=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_ch, 1, 1))
            self._bias_is_param = True
        else:
            self.register_buffer("bias", torch.zeros(out_ch, 1, 1))
            self._bias_is_param = False

        self.h_out = h_out
        self.w_out = w_out

        if weight_gain is not None:
            kH = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            kW = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
            size_pre = in_ch * kH * kW
            scale = weight_gain * (1.0 / size_pre) ** 0.5
            nn.init.uniform_(self.conv.weight, -scale, +scale)
        if bias and bias_gain is not None:
            nn.init.uniform_(self.bias, -bias_gain, +bias_gain)

        s = self.conv.stride
        if isinstance(s, int):
            self._stride_tuple = (s, s)
        else:
            self._stride_tuple = tuple(s)

    def create_state(self, batch_size, device):
        state = torch.zeros(batch_size, self.conv.out_channels, self.h_out, self.w_out,
                            device=device, requires_grad=False).contiguous(memory_format=torch.channels_last)
        return state

    def energy(self, pre, post):
        feat = self.conv(pre)
        feat_pooled = F.max_pool2d(feat, 2)
        feat_pooled = feat_pooled.contiguous(memory_format=torch.channels_last)
        e = -(feat_pooled * post).sum(dim=(1, 2, 3))
        bias = self.bias.view(1, -1, 1, 1)
        e += -(bias * post).sum(dim=(1, 2, 3))
        return e

    def backward(self, pre, post):
        with torch.no_grad():
            feat = self.conv(pre)
            # pool with indices to be able to unpool deterministically
            feat_pooled, indices = F.max_pool2d(feat, 2, return_indices=True)
            feat_pooled = feat_pooled.contiguous(memory_format=torch.channels_last)

            bias = self.bias.view(1, -1, 1, 1)
            grad_post = -(feat_pooled + bias).contiguous(memory_format=torch.channels_last)

            # unpool into the original feat shape
            post_unpool = F.max_unpool2d(post, indices, kernel_size=2, stride=2, output_size=feat.shape)
            post_unpool = post_unpool.contiguous(memory_format=torch.channels_last)

            output_padding = (self._stride_tuple[0] - 1, self._stride_tuple[1] - 1)
            grad_pre = -F.conv_transpose2d(
                input=post_unpool,
                weight=self.conv.weight,
                stride=self.conv.stride,
                padding=self.conv.padding,
                output_padding=output_padding
            ).contiguous(memory_format=torch.channels_last)

        return grad_post, grad_pre


class InteractionLinear(Interaction):
    def __init__(self, in_features, out_features, bias=True, weight_gain=None, bias_gain=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
            self._bias_is_param = True
        else:
            self.register_buffer("bias", torch.zeros(out_features))
            self._bias_is_param = False

        if weight_gain is not None:
            size_pre = in_features
            scale = weight_gain * (1.0 / size_pre) ** 0.5
            nn.init.uniform_(self.linear.weight, -scale, +scale)
        if bias and bias_gain is not None:
            nn.init.uniform_(self.bias, -bias_gain, +bias_gain)


    def create_state(self, batch_size, device):
        return torch.zeros(batch_size, self.linear.out_features, device=device, requires_grad=False)

    def energy(self, pre, post):
        if pre.dim() > 2:
            pre_flat = pre.reshape(pre.size(0), -1).contiguous()
        else:
            pre_flat = pre
        proj = self.linear(pre_flat)
        e = -(proj * post).sum(dim=1)
        bias = self.bias.view(1, -1)
        e += -(bias * post).sum(dim=1)
        return e

    def backward(self, pre, post):
        with torch.no_grad():
            if pre.dim() > 2:
                pre_flat = pre.reshape(pre.size(0), -1).contiguous()
            else:
                pre_flat = pre

            grad_post = -self.linear(pre_flat)
            grad_post = grad_post - self.bias.view(1, -1)

            grad_pre_flat = -torch.matmul(post, self.linear.weight)
            grad_pre = grad_pre_flat.reshape(pre.shape)
        return grad_post, grad_pre
