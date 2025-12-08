import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from .layers import EqPropConv2d, EqPropLinear, ResBlock, EqPropConvMaxPool2d, identity
from .functional import compute_betas

torch.set_float32_matmul_precision('high')

class BaseHopfieldModel(nn.Module):
    def __init__(self):
        super().__init__()

    def create_states(self, batch_size, device):
        return [layer.create_state(batch_size, device) for layer in self.layers]

    def energy(self, x, states, beta=0.0, target=None, streams=None):
        E = 0
        inputs = [x] + [states[i][-1] for i in range(len(self.layers))]
        if streams is None:
            for i, layer in enumerate(self.layers):
                with torch.cuda.nvtx.range(f"Layer {i} Energy"):
                    E += layer.energy(inputs[i], states[i])
        else:
            energies = [None] * len(self.layers)
            torch.cuda.synchronize()
            for i, layer in enumerate(self.layers):
                stream = streams[i]
                with torch.cuda.stream(stream):
                    with torch.cuda.nvtx.range(f"Layer {i} Energy"):
                        energies[i] = layer.energy(inputs[i], states[i])
            torch.cuda.synchronize()
        
            E = torch.stack(energies).sum(0)


        if beta != 0.0 and target is not None:
            with torch.cuda.nvtx.range("Cost Energy Calculation"):
                logits = states[-1][-1]  # Output state is always last element of last layer
                if self.cost_type == 'CE':
                    # Expect target as class indices of shape (B,)
                    log_probs = F.log_softmax(logits, dim=1)
                    E += beta * (-(log_probs.gather(1, target.view(-1, 1)).squeeze(1)))
                else:
                    # MSE on logits vs one-hot target; accept either class indices or one-hot
                    if target.dim() == 1:
                        one_hot = F.one_hot(target, num_classes=logits.shape[1]).float()
                    else:
                        one_hot = target.float()
                    E += beta * 0.5 * ((logits - one_hot) ** 2).sum(dim=1)
        return E

    @torch.no_grad()
    def minimize_step(self, x, states, beta=0.0, target=None, streams=None):
        n_layers = len(self.layers)
        pre_grads = [None] * n_layers
        post_grads = [None] * n_layers
        # Build inputs list: [x, output_state_0, output_state_1, ...]
        # All layers return list of states; output is always last element
        # Run all layer backwards, in serial or parallel
        inputs = [x] + [states[i][-1] for i in range(n_layers)]

        if streams is None:
            for i, layer in enumerate(self.layers):
                torch.cuda.nvtx.range_push(f"Layer {i}")
                grad_states, grad_pre = layer.backward(inputs[i], states[i])
                torch.cuda.nvtx.range_pop()
                if i != 0:
                    pre_grads[i-1] = grad_pre
                post_grads[i] = grad_states
            output_state = states[-1][-1]
            pre_grads[-1] = torch.zeros_like(output_state) # last layer is always linear 
        else:
            torch.cuda.synchronize()      
            # Launch all backward passes in different streams
            for i, layer in enumerate(self.layers):
                stream = streams[i]
                with torch.cuda.stream(stream):
                    torch.cuda.nvtx.range_push(f"Layer {i}")
                    grad_states, grad_pre = layer.backward(inputs[i], states[i])
                    torch.cuda.nvtx.range_pop()
                    if i != 0:
                        pre_grads[i-1] = grad_pre
                    post_grads[i] = grad_states
            torch.cuda.synchronize()
            
            output_state = states[-1][-1]
            pre_grads[-1] = torch.zeros_like(output_state) # last layer is always linear

        # Combine post and pre grads - all post_grads are lists, pre_grad applies to last element
        state_grads = []
        for i in range(n_layers):
            post_grad_list = post_grads[i]  # Always a list
            pre_grad = pre_grads[i]
            # pre_grad applies to last element of the state list
            combined = list(post_grad_list)  # Copy list
            if pre_grad is not None:
                combined[-1] = combined[-1] + pre_grad
            state_grads.append(combined)
        
        # Add cost on output layer if needed
        if beta != 0.0 and target is not None:
            logits = states[-1][-1]  # Output state is always last element of last layer
            if self.cost_type == 'CE':
                cost_grad = F.softmax(logits, dim=1) - F.one_hot(target, num_classes=logits.shape[1]).float()
            else: # MSE
                one_hot_target = F.one_hot(target, num_classes=logits.shape[1]).float()
                cost_grad = logits - one_hot_target
            # Apply cost to the last element of state_grads[-1]
            state_grads[-1][-1] = state_grads[-1][-1] + beta * cost_grad
        
        # Update states using layer.update_state
        new_states = []
        for i, layer in enumerate(self.layers):
            # Pass the gradients for this layer's states
            new_state_list = layer.update_state(state_grads[i])
            new_states.append(new_state_list)
            
        return new_states

    def minimize(self, x, states, beta=0.0, target=None, n_iters=20, streams=None):
        current_states = list(states)

        for cnt in range(n_iters):
            torch.cuda.nvtx.range_push(f"[{cnt}] Minimize Step")
            current_states = self.minimize_step(x, current_states, beta, target, streams)
            torch.cuda.nvtx.range_pop()
        return current_states


class ConvHopfieldEnergy32(BaseHopfieldModel):
    def __init__(self):
        super().__init__()
        weight_gains = [0.4, 0.7, 0.6, 0.3, 0.4]
        bias_gains   = [
            0.5 / ( (3   * 3 * 3) ** 0.5 ),
            0.5 / ( (128 * 3 * 3) ** 0.5 ),
            0.5 / ( (256 * 3 * 3) ** 0.5 ),
            0.5 / ( (512 * 3 * 3) ** 0.5 ),
            0.5 / ( (512 * 2 * 2) ** 0.5 ),
        ]
        self.layers = nn.ModuleList([
            EqPropConvMaxPool2d(3,   128, 3, h_out=16, w_out=16, padding=1, weight_gain=weight_gains[0], bias_gain=bias_gains[0]),
            EqPropConvMaxPool2d(128, 256, 3, h_out=8,  w_out=8,  padding=1, weight_gain=weight_gains[1], bias_gain=bias_gains[1]),
            EqPropConvMaxPool2d(256, 512, 3, h_out=4,  w_out=4,  padding=1, weight_gain=weight_gains[2], bias_gain=bias_gains[2]),
            EqPropConvMaxPool2d(512, 512, 3, h_out=2,  w_out=2,  padding=1, weight_gain=weight_gains[3], bias_gain=bias_gains[3]),
            EqPropLinear(512*2*2, 10, weight_gain=weight_gains[4], bias_gain=bias_gains[4], activation=identity)
        ])
        self.output = self.layers[-1]
        self.cost_type = 'CE'

        self.reinit_like_original() # for reproducibility

    def reinit_like_original(self, seed: int = 0):
        """Reinitialize parameters to exactly match original main.py draw order.
        Order: all biases (b1..b5), then conv weights (W1..W4), then dense W5 drawn
        as (512,2,2,10) and permuted to (10,2048).
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Same gains as used at construction
        bias_gains = [
            0.5 / ( (3   * 3 * 3) ** 0.5 ),
            0.5 / ( (128 * 3 * 3) ** 0.5 ),
            0.5 / ( (256 * 3 * 3) ** 0.5 ),
            0.5 / ( (512 * 3 * 3) ** 0.5 ),
            0.5 / ( (512 * 2 * 2) ** 0.5 ),
        ]
        weight_gains = [0.4, 0.7, 0.6, 0.3, 0.4]

        # Unpack layers
        conv1: EqPropConv2d = self.layers[0]
        conv2: EqPropConv2d = self.layers[1]
        conv3: EqPropConv2d = self.layers[2]
        conv4: EqPropConv2d = self.layers[3]
        fc:   EqPropLinear  = self.layers[4]

        # 1) Biases first (b1..b5)
        with torch.no_grad():
            torch.nn.init.uniform_(conv1.bias, -bias_gains[0], +bias_gains[0])
            torch.nn.init.uniform_(conv2.bias, -bias_gains[1], +bias_gains[1])
            torch.nn.init.uniform_(conv3.bias, -bias_gains[2], +bias_gains[2])
            torch.nn.init.uniform_(conv4.bias, -bias_gains[3], +bias_gains[3])
            torch.nn.init.uniform_(fc.bias,    -bias_gains[4], +bias_gains[4])

        # 2) Conv weights (W1..W4) with kaiming_uniform-style scale = gain*sqrt(1/size_pre)
        with torch.no_grad():
            for gain, layer in zip(weight_gains[:4], [conv1, conv2, conv3, conv4]):
                kH = layer.conv.weight.shape[2]
                kW = layer.conv.weight.shape[3]
                size_pre = layer.conv.in_channels * kH * kW
                scale = gain * (1.0 / size_pre) ** 0.5
                torch.nn.init.uniform_(layer.conv.weight, -scale, +scale)

        # 3) Dense weight: draw in original (512,2,2,10), then permute to (10,2048)
        with torch.no_grad():
            size_pre_fc = 512 * 2 * 2
            scale_fc = weight_gains[4] * (1.0 / size_pre_fc) ** 0.5
            tmp = torch.empty(512, 2, 2, 10, device=fc.linear.weight.device, dtype=fc.linear.weight.dtype)
            torch.nn.init.uniform_(tmp, -scale_fc, +scale_fc)
            w_lin = tmp.permute(3, 0, 1, 2).contiguous().view(10, 2048)
            fc.linear.weight.copy_(w_lin)

class ResNet13(BaseHopfieldModel):
    """ResNet13 model for 32x32 input images with 4 residual blocks followed by a dense layer.
    """
    def __init__(self, num_inputs=3, num_outputs=10, num_hiddens_1=128, num_hiddens_2=256, 
                 num_hiddens_3=512, num_hiddens_4=512, activation='hard-sigmoid',
                 ):
        super().__init__()
        
        # Calculate bias gains based on reference pattern
        bias_gains = [
            0.5 / np.sqrt(num_inputs * 3 * 3),      # Block 1 conv1
            0.5 / np.sqrt(num_hiddens_1 * 3 * 3),  # Block 1 conv2
            0.5 / np.sqrt(num_inputs * 1 * 1),     # Block 1 skip
            0.5 / np.sqrt(num_hiddens_1 * 3 * 3),  # Block 2 conv1
            0.5 / np.sqrt(num_hiddens_2 * 3 * 3),  # Block 2 conv2
            0.5 / np.sqrt(num_hiddens_1 * 1 * 1),  # Block 2 skip
            0.5 / np.sqrt(num_hiddens_2 * 3 * 3),  # Block 3 conv1
            0.5 / np.sqrt(num_hiddens_3 * 3 * 3),  # Block 3 conv2
            0.5 / np.sqrt(num_hiddens_2 * 1 * 1),  # Block 3 skip
            0.5 / np.sqrt(num_hiddens_3 * 3 * 3),  # Block 4 conv1
            0.5 / np.sqrt(num_hiddens_4 * 3 * 3),  # Block 4 conv2
            0.5 / np.sqrt(num_hiddens_3 * 1 * 1),  # Block 4 skip
            0.5 / np.sqrt(num_hiddens_4 * 2 * 2),  # Dense layer
        ]
        weight_gains = [
                    0.6, 0.6, 0.7,   # block-1 (skip stronger)
                    0.6, 0.7, 0.6,   # block-2
                    0.6, 0.7, 0.6,   # block-3
                    0.6, 0.7, 0.6,   # block-4
                    0.8      
        ]
        
        self.layers = nn.ModuleList([
            # Block 1: 3 -> 128, 32x32 -> 16x16
            ResBlock(num_inputs, num_hiddens_1, h_out=16, w_out=16,
                    strides=[1,2,2],
                    h_intermediate=32, w_intermediate=32,
                    weight_gains=weight_gains[0:3],
                    bias_gains=bias_gains[0:3]),
            # Block 2: 128 -> 256, 16x16 -> 8x8
            ResBlock(num_hiddens_1, num_hiddens_2, h_out=8, w_out=8,
                    strides=[1,2,2],
                    h_intermediate=16, w_intermediate=16,
                    weight_gains=weight_gains[3:6],
                    bias_gains=bias_gains[3:6]),
            # Block 3: 256 -> 512, 8x8 -> 4x4
            ResBlock(num_hiddens_2, num_hiddens_3, h_out=4, w_out=4,
                    strides=[1,2,2],
                    h_intermediate=8, w_intermediate=8,
                    weight_gains=weight_gains[6:9],
                    bias_gains=bias_gains[6:9]),
            # Block 4: 512 -> 512, 4x4 -> 2x2
            ResBlock(num_hiddens_3, num_hiddens_4, h_out=2, w_out=2,
                    strides=[1,2,2],
                    h_intermediate=4, w_intermediate=4,
                    weight_gains=weight_gains[9:12],
                    bias_gains=bias_gains[9:12]),
            # Dense layer: 512*2*2 -> num_outputs
            EqPropLinear(num_hiddens_4 * 2 * 2, num_outputs,
                        weight_gain=weight_gains[12], bias_gain=bias_gains[12], activation=identity)
        ])
        self.output = self.layers[-1]
        self.cost_type = 'CE'


class ResNet16(BaseHopfieldModel):
    """ResNet16 model for 32x32 input images with 5 residual blocks followed by a dense layer.
    """
    def __init__(self, num_inputs=3, num_outputs=10, num_hiddens_1=128, num_hiddens_2=256, 
                 num_hiddens_3=512, num_hiddens_4=512, num_hiddens_5=1024,
                 ):
        super().__init__()
        
        # Calculate bias gains based on reference pattern
        bias_gains = [
            0.5 / np.sqrt(num_inputs * 3 * 3),      # Block 1 conv1
            0.5 / np.sqrt(num_hiddens_1 * 3 * 3),  # Block 1 conv2
            0.5 / np.sqrt(num_inputs * 1 * 1),     # Block 1 skip
            0.5 / np.sqrt(num_hiddens_1 * 3 * 3),  # Block 2 conv1
            0.5 / np.sqrt(num_hiddens_2 * 3 * 3),  # Block 2 conv2
            0.5 / np.sqrt(num_hiddens_1 * 1 * 1),  # Block 2 skip
            0.5 / np.sqrt(num_hiddens_2 * 3 * 3),  # Block 3 conv1
            0.5 / np.sqrt(num_hiddens_3 * 3 * 3),  # Block 3 conv2
            0.5 / np.sqrt(num_hiddens_2 * 1 * 1),  # Block 3 skip
            0.5 / np.sqrt(num_hiddens_3 * 3 * 3),  # Block 4 conv1
            0.5 / np.sqrt(num_hiddens_4 * 3 * 3),  # Block 4 conv2
            0.5 / np.sqrt(num_hiddens_3 * 1 * 1),  # Block 4 skip
            0.5 / np.sqrt(num_hiddens_3 * 3 * 3),  # Block 5 conv1
            0.5 / np.sqrt(num_hiddens_4 * 3 * 3),  # Block 5 conv2
            0.5 / np.sqrt(num_hiddens_3 * 1 * 1),  # Block 5 skip
            0.5 / np.sqrt(num_hiddens_4 * 2 * 2),  # Dense layer
        ]

        weight_gains = [
                    0.6, 0.6, 0.7,   # block-1 (skip stronger)
                    0.6, 0.7, 0.6,   # block-2
                    0.6, 0.7, 0.6,   # block-3
                    0.6, 0.7, 0.6,   # block-4
                    0.6, 0.7, 0.6,   # block-5
                    0.8      
        ]
        
        self.layers = nn.ModuleList([
            # Block 1: 3 -> 128, 32x32 -> 16x16
            ResBlock(num_inputs, num_hiddens_1, h_out=16, w_out=16,
                    strides=[1,2,2],
                    h_intermediate=32, w_intermediate=32,
                    weight_gains=weight_gains[0:3],
                    bias_gains=bias_gains[0:3]),
            # Block 2: 128 -> 256, 16x16 -> 8x8
            ResBlock(num_hiddens_1, num_hiddens_2, h_out=8, w_out=8,
                    strides=[1,2,2],
                    h_intermediate=16, w_intermediate=16,
                    weight_gains=weight_gains[3:6],
                    bias_gains=bias_gains[3:6]),
            # Block 3: 256 -> 512, 8x8 -> 4x4
            ResBlock(num_hiddens_2, num_hiddens_3, h_out=4, w_out=4,
                    strides=[1,2,2],
                    h_intermediate=8, w_intermediate=8,
                    weight_gains=weight_gains[6:9],
                    bias_gains=bias_gains[6:9]),
            # Block 4: 512 -> 512, 4x4 -> 2x2
            ResBlock(num_hiddens_3, num_hiddens_4, h_out=2, w_out=2,
                    strides=[1,2,2],
                    h_intermediate=4, w_intermediate=4,
                    weight_gains=weight_gains[9:12],
                    bias_gains=bias_gains[9:12]),
            # Block 5: 512 -> 512, 2x2 -> 1x1
            ResBlock(num_hiddens_4, num_hiddens_5, h_out=2, w_out=2,
                    strides=[1,1,1],
                    h_intermediate=2, w_intermediate=2,
                    weight_gains=weight_gains[12:15],
                    bias_gains=bias_gains[12:15]),
            # Dense layer: 512*1*1 -> num_outputs
            EqPropLinear(num_hiddens_5 * 2 * 2, num_outputs,
                        weight_gain=weight_gains[15], bias_gain=bias_gains[15], activation=identity)
        ])
        self.output = self.layers[-1]
        self.cost_type = 'CE'


def train_batch_centered(model, x, y, optimizer, beta=0.1, use_mean_reduction=True, n_iters_free=50, n_iters_nudged=50, streams=None, previous_states=None):
    B = x.size(0)
    device = x.device
    
    if previous_states is not None:
        if previous_states[0][0].size(0) == B:
             states = previous_states
        else:
             states = model.create_states(B, device)
    else:
        states = model.create_states(B, device)
    
    # Free phase
    free_states = model.minimize(x, states, beta=0.0, n_iters=n_iters_free, streams=streams)
    free_states = [[s.detach() for s in state_list] for state_list in free_states]
    logits_free = free_states[-1][-1]

    # Centered nudging
    b1, b2, denom = compute_betas('centered', beta)
    
    # Phase 1
    nudged_states = model.minimize(x, free_states, beta=b1, target=y, n_iters=n_iters_nudged, streams=streams)
    nudged_states = [[s.detach() for s in state_list] for state_list in nudged_states]
    E_1 = model.energy(x, nudged_states, b1, target=y)
    E_1 = E_1.mean() if use_mean_reduction else E_1.sum()
    with torch.cuda.nvtx.range("Weight Grads 1"):
        grads_1 = torch.autograd.grad(E_1, model.parameters(), create_graph=False)
    
    # Phase 2 (restart from free)
    nudged_states = model.minimize(x, free_states, beta=b2, target=y, n_iters=n_iters_nudged, streams=streams)
    nudged_states = [[s.detach() for s in state_list] for state_list in nudged_states]
    E_2 = model.energy(x, nudged_states, b2, target=y)
    E_2 = E_2.mean() if use_mean_reduction else E_2.sum()
    with torch.cuda.nvtx.range("Weight Grads 2"):
        grads_2 = torch.autograd.grad(E_2, model.parameters(), create_graph=False)
    # EP update
    with torch.cuda.nvtx.range("Optimizer Step"):
        grads = [((g2 - g1).detach() / denom) for g1, g2 in zip(grads_1, grads_2)]
        optimizer.zero_grad()
        for p, g in zip(model.parameters(), grads):
            p.grad = g
        optimizer.step()
    
    # Task loss from free-phase logits
    if model.cost_type == 'CE':
        batch_loss = F.cross_entropy(logits_free, y).item()
    else:
        one_hot = F.one_hot(y, num_classes=logits_free.shape[1]).float()
        batch_loss = (0.5 * ((logits_free - one_hot) ** 2).sum(dim=1)).mean().item()
    return float(E_1.item()), float(E_2.item()), logits_free, batch_loss, free_states


def evaluate(model, dataloader, device: str, n_iters_infer: int = 120, streams=None):
    model.eval()
    total = 0
    correct = 0
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        B = x.size(0)
        states = model.create_states(B, device)
        free_states = model.minimize(x, states, beta=0.0, n_iters=n_iters_infer, streams=streams)
        logits = free_states[-1][-1]
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += B
    return correct / max(1, total)
