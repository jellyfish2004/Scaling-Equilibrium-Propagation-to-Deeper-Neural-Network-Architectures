import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import MaxUnpool2d
from .activation import LearnableClampedReLU, hard_sigmoid, identity

class EqPropLayer(nn.Module):
    def __init__(self, activation=None):
        super().__init__()
        self.activation = activation if activation is not None else hard_sigmoid
    
    def create_state(self, batch_size, device):
        raise NotImplementedError
    
    def energy(self, pre, state):
        raise NotImplementedError

    def backward(self, pre, state):
        raise NotImplementedError

    def update_state(self, grads):
        """
        Update state based on gradients (dE/ds).
        Default implementation for single-state layers: s = activation(-dE/ds).
        """
        # grads is a list of gradients for the states managed by this layer
        return [self.activation(-grads[0])]

class EqPropConvMaxPool2d(EqPropLayer):
    def __init__(self, in_ch, out_ch, kernel_size, h_out, w_out, stride=1, padding=0, bias=True, weight_gain=None, bias_gain=None, activation=None):
        super().__init__(activation=activation)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_ch,1,1)) if bias else None
        self.h_out = h_out
        self.w_out = w_out

        if weight_gain is not None:
            kH = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            kW = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
            size_pre = in_ch * kH * kW
            scale = weight_gain * (1.0 / size_pre) ** 0.5
            nn.init.uniform_(self.conv.weight, -scale, +scale)
        if self.bias is not None and bias_gain is not None:
            nn.init.uniform_(self.bias, -bias_gain, +bias_gain)

    def create_state(self, batch_size, device):
        state = torch.zeros(batch_size, self.conv.out_channels, self.h_out, self.w_out, device=device, requires_grad=False).to(memory_format=torch.channels_last)
        return [state]

    def energy(self, pre, states):
        state = states[0]  # Single-state layers: extract first (and only) state
        feat = self.conv(pre)
        feat = F.max_pool2d(feat, 2).to(memory_format=torch.channels_last)
        e = -(feat * state).sum(dim=(1,2,3))
        if self.bias is not None:
            e += -(self.bias * state).sum(dim=(1,2,3))
        return e

    def backward(self, pre, states):
        # Extract state - states is always a list, extract first element
        state = states[0]
        
        with torch.no_grad(): # No need to track gradients for this calculation
            feat = self.conv(pre)
            feat_pooled, indices = F.max_pool2d(feat, 2, return_indices=True)
            
            grad_state = -(feat_pooled + self.bias) if self.bias is not None else -feat_pooled
            state_unpool = F.max_unpool2d(state, indices, 2, output_size=feat.shape)

            if self.conv.stride == (2, 2): # hardcoding for now, since we have only 2 possible cases currently
                output_padding = (1, 1)
            else:
                output_padding = (0, 0)
            
            grad_pre = - F.conv_transpose2d(
                input=state_unpool,
                weight=self.conv.weight,
                stride=self.conv.stride,
                padding=self.conv.padding,
                output_padding=output_padding
            )

        return [grad_state], grad_pre  # Always return list for uniform interface

class EqPropConv2d(EqPropLayer):
    def __init__(self, in_ch, out_ch, kernel_size, h_out, w_out, stride=1, padding=0, bias=True, weight_gain=None, bias_gain=None, activation=None):
        super().__init__(activation=activation)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_ch,1,1)) if bias else None
        self.h_out = h_out
        self.w_out = w_out

        if weight_gain is not None:
            kH = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
            kW = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
            size_pre = in_ch * kH * kW
            scale = weight_gain * (1.0 / size_pre) ** 0.5
            nn.init.uniform_(self.conv.weight, -scale, +scale)
        if self.bias is not None and bias_gain is not None:
            nn.init.uniform_(self.bias, -bias_gain, +bias_gain)

    def create_state(self, batch_size, device):
        state = torch.zeros(batch_size, self.conv.out_channels, self.h_out, self.w_out, device=device, requires_grad=False).to(memory_format=torch.channels_last)
        return [state]  # Always return list for uniform interface

    def energy(self, pre, states):
        state = states[0]  # Single-state layers: extract first (and only) state
        feat = self.conv(pre)
        e = -(feat * state).sum(dim=(1,2,3))
        if self.bias is not None:
            e += -(self.bias * state).sum(dim=(1,2,3))
        return e

    def backward(self, pre, states):
        # Extract state - states is always a list, extract first element
        state = states[0]
        
        with torch.no_grad(): # No need to track gradients for this calculation
            feat = self.conv(pre)
            grad_state = -(feat + self.bias).to(memory_format=torch.channels_last) if self.bias is not None else -feat.to(memory_format=torch.channels_last)
            if self.conv.stride == (2, 2): # hardcoding for now, since we have only 2 possible cases currently
                output_padding = (1, 1)
            else:
                output_padding = (0, 0)
            
            grad_pre = - F.conv_transpose2d(
                input=state,
                weight=self.conv.weight,
                stride=self.conv.stride,
                padding=self.conv.padding,
                output_padding=output_padding
            ).to(memory_format=torch.channels_last)
        return [grad_state], grad_pre  # Always return list for uniform interface

class EqPropLinear(EqPropLayer):
    def __init__(self, in_features, out_features, bias=True, weight_gain=None, bias_gain=None, activation=None):
        super().__init__(activation=activation)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        # CHANGE: move weight/bias initialization into the layer with optional gains
        if weight_gain is not None:
            size_pre = in_features
            scale = weight_gain * (1.0 / size_pre) ** 0.5
            nn.init.uniform_(self.linear.weight, -scale, +scale)
        if self.bias is not None and bias_gain is not None:
            nn.init.uniform_(self.bias, -bias_gain, +bias_gain)

    def create_state(self, batch_size, device):
        state = torch.zeros(batch_size, self.linear.out_features, device=device, requires_grad=False)
        return [state]  # Always return list for uniform interface

    def energy(self, pre, states):
        state = states[0]  # Single-state layers: extract first (and only) state
        if pre.dim() > 2: # B, C, W, H
            # Use reshape to handle non-contiguous (channels_last) tensors
            pre = pre.reshape(pre.size(0), -1)
        proj = self.linear(pre)
        e = -(proj * state).sum(dim=1)
        if self.bias is not None:
            e += -(self.bias * state).sum(dim=1)
        return e

    def backward(self, pre, states):
        state = states[0]  # Single-state layers: extract first (and only) state
        with torch.no_grad():
            if pre.dim() > 2:
                pre_flat = pre.reshape(pre.size(0), -1).contiguous()
            else:
                pre_flat = pre
            grad_state = -self.linear(pre_flat) - self.bias
            grad_pre_flat = -torch.matmul(state, self.linear.weight)
            grad_pre = grad_pre_flat.reshape(pre.shape)
        return [grad_state], grad_pre  # Always return list for uniform interface

class ResBlock(EqPropLayer):
    """
    Residual Block for Equilibrium Propagation.
    
    Architecture:
    - Main path: Conv3x3 -> s^n -> Conv3x3 -> output
    - Residual path: Conv1x1 -> output (no bias)
    - Output: main_path + residual_path
    
    The block manages two internal states:
    - s_n: intermediate state after first Conv3x3
    - s_np1: output state (after second Conv3x3 + residual)
    """
    def __init__(self, in_ch, out_ch, h_out, w_out, strides=[1,2,2], 
                 h_intermediate=None, w_intermediate=None, padding=1,
                 weight_gains=[None, None, None],
                 bias_gains=[None, None],  # Only 2 biases: conv1, conv2 (skip has no bias)
                 activation=None):
        super().__init__(activation=activation)
        
        stride_conv1, stride_conv2, stride_skip = strides
        weight_gain_conv1, weight_gain_conv2, weight_gain_skip = weight_gains
        bias_gain_conv1, bias_gain_conv2 = bias_gains  # Only 2 bias gains
        # Calculate intermediate dimensions if not provided
        if h_intermediate is None:
            # Assume stride_conv1=1 means same size, otherwise halve
            h_intermediate = h_out if stride_conv1 == 1 else h_out * 2
            w_intermediate = w_out if stride_conv1 == 1 else w_out * 2

        if isinstance(activation, list):
            assert len(activation) == 2, "Activation must be a list of length 2 for ResBlock"
            activation_conv1, activation_conv2 = activation
        else:
            activation_conv1 = activation_conv2 = activation
        
        # Main path: two 3x3 convolutions
        self.conv1 = EqPropConv2d(in_ch, out_ch, 3, h_intermediate, w_intermediate, 
                                   stride=stride_conv1, padding=padding,
                                   weight_gain=weight_gain_conv1, bias_gain=bias_gain_conv1,
                                   activation=activation_conv1)
        
        # Second conv with potentially different stride
        self.conv2 = EqPropConv2d(out_ch, out_ch, 3, h_out, w_out,
                                   stride=stride_conv2, padding=padding,
                                   weight_gain=weight_gain_conv2, bias_gain=bias_gain_conv2,
                                   activation=activation_conv2)
        
        # Residual path: 1x1 convolution for dimension matching (NO BIAS)
        self.conv_skip = EqPropConv2d(in_ch, out_ch, 1, h_out, w_out,
                                       stride=stride_skip, padding=0,
                                       weight_gain=weight_gain_skip, bias=False,
                                       activation=activation_conv2)
        
        self.h_out = h_out
        self.w_out = w_out
        self.out_ch = out_ch
    
    def create_state(self, batch_size, device):
        """
        Returns states for the ResBlock:
        - state[0]: intermediate state s^n (after conv1)
        - state[1]: output state s^(n+1) (final output)
        """
        # Sub-layers return lists, extract the actual tensors
        s_n = self.conv1.create_state(batch_size, device)[0]
        s_np1 = self.conv2.create_state(batch_size, device)[0]
        return [s_n, s_np1]
    
    def energy(self, pre, states):
        s_n, s_np1 = states[0], states[1]  # Extract states from list
        E_conv1 = self.conv1.energy(pre, [s_n])  # Wrap in list for sub-layer
        E_conv2 = self.conv2.energy(s_n, [s_np1])  # Wrap in list for sub-layer
        E_residual = self.conv_skip.energy(pre, [s_np1])  # Wrap in list for sub-layer
        
        return E_conv1 + E_conv2 + E_residual
    
    def backward(self, pre, states):
        s_n, s_np1 = states[0], states[1]
        # conv1: pre -> s_n
        grad_states_conv1, grad_pre_from_conv1 = self.conv1.backward(pre, [s_n])
        grad_s_n_from_conv1 = grad_states_conv1[0]
        
        # conv2: s_n -> s_np1
        grad_states_conv2, grad_s_n_from_conv2 = self.conv2.backward(s_n, [s_np1])
        grad_s_np1_from_conv2 = grad_states_conv2[0]
        
        # conv_skip: pre -> s_np1
        grad_states_skip, grad_pre_from_skip = self.conv_skip.backward(pre, [s_np1])
        grad_s_np1_from_skip = grad_states_skip[0]
        
        grad_s_np1 = (grad_s_np1_from_conv2 + grad_s_np1_from_skip).to(memory_format=torch.channels_last)  # both conv2 and skip contribute to s_np1
        grad_s_n = (grad_s_n_from_conv1 + grad_s_n_from_conv2).to(memory_format=torch.channels_last)  # both conv1 and conv2 affect s_n
        grad_pre = (grad_pre_from_conv1 + grad_pre_from_skip).to(memory_format=torch.channels_last)  # both conv1 and skip affect pre
        
        return [grad_s_n, grad_s_np1], grad_pre  # Always return list for uniform interface

    def update_state(self, grads):
        # grads = [grad_s_n, grad_s_np1]
        # s_n uses conv1's activation
        s_n = self.conv1.activation(-grads[0])
        # s_np1 uses conv2's activation (which should be same as conv_skip's)
        s_np1 = self.conv2.activation(-grads[1])
        return [s_n, s_np1]
