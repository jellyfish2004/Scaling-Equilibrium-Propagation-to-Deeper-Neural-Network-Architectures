import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import inspect
from tqdm import tqdm
from eqprop.functional import compute_betas
from eqprop.activation import hard_sigmoid, identity
from .layers import InteractionConv2d, InteractionConvMaxPool2d, InteractionLinear

class InteractionBaseHopfieldModel(nn.Module):
    def __init__(self, state_shapes, interactions, connections, activations):
        super().__init__()
        self.n_states = len(state_shapes)
        self.state_shapes = state_shapes
        self.interactions = nn.ModuleList(interactions)
        self.connections = connections  # [(interaction, (src_idx, dst_idx)), ...]
        self.activations = activations  # list of functions for each state

        if len(self.activations) != self.n_states:
            raise ValueError("Number of activations must match number of states")

        self.cost_type = 'MSE'  # default

    def create_states(self, batch_size, device):
        states = [None] * self.n_states
        for interaction, link in self.connections:
            idx = link[1] - 1
            if 0 <= idx < self.n_states:
                if states[idx] is None:
                    states[idx] = interaction.create_state(batch_size, device)
        # fill any remaining None states with zeros (defensive)
        for i in range(len(states)):
            if states[i] is None:
                shape = self.state_shapes[i]
                device = states[0].device if states[0] is not None else torch.device("cpu")
                states[i] = torch.zeros((batch_size, *shape), device=device, requires_grad=False)
        return states

    def energy(self, x, states, beta=0.0, target=None):
        E = 0
        inputs = [x] + states
        for i, (interaction, link) in enumerate(self.connections):
            E = E + interaction.energy(inputs[link[0]], inputs[link[1]])
        if beta != 0.0 and target is not None:
            logits = states[-1]
            if self.cost_type == 'CE':
                log_probs = F.log_softmax(logits, dim=1)
                E = E + beta * (-(log_probs.gather(1, target.view(-1, 1)).squeeze(1)))
            else:
                if target.dim() == 1:
                    one_hot = F.one_hot(target, num_classes=logits.shape[1]).float()
                else:
                    one_hot = target.float()
                diff = logits - one_hot
                E = E + beta * 0.5 * (diff * diff).sum(dim=1)
        return E

    @torch.no_grad()
    def minimize_step(self, x, states, beta=0.0, target=None, streams=None):
        pre_grads = [None] * len(self.connections)
        post_grads = [None] * len(self.connections)
        inputs = [x] + states
        state_grads = [torch.zeros_like(s) for s in inputs] # incl input here

        for i, (interaction, link) in enumerate(self.connections):
            grad_post, grad_pre = interaction.backward(inputs[link[0]], inputs[link[1]])
            pre_grads[i] = grad_pre
            post_grads[i] = grad_post

        for i, (interaction, link) in enumerate(self.connections):
            state_grads[link[0]] = state_grads[link[0]] + pre_grads[i]
            state_grads[link[1]] = state_grads[link[1]] + post_grads[i]

        state_grads = state_grads[1:] # remove input

        # add cost gradient on output state if nudged
        if beta != 0.0 and target is not None:
            logits = states[-1]
            if self.cost_type == 'CE':
                cost_grad = F.softmax(logits, dim=1) - F.one_hot(target, num_classes=logits.shape[1]).float()
            else:
                one_hot_target = F.one_hot(target, num_classes=logits.shape[1]).float()
                cost_grad = logits - one_hot_target
            state_grads[-1] = state_grads[-1] + beta * cost_grad

        # Clamp and update with activation functions
        new_states = []
        for i, grad in enumerate(state_grads):
            activation_fn = self.activations[i]
            new_states.append(activation_fn(-grad))
        return new_states

    def minimize(self, x, states, beta=0.0, target=None, n_iters=20, streams=None):
        current_states = list(states)
        for _ in range(n_iters):
            current_states = self.minimize_step(x, current_states, beta, target, streams)
            current_states = [s.clone() for s in current_states]
        return current_states


class ConvHopfieldEnergy32_Interactions(InteractionBaseHopfieldModel):
    def __init__(self):
        weight_gains = [0.4, 0.7, 0.6, 0.3, 0.4]
        bias_gains = [
            0.5 / ((3   * 3 * 3) ** 0.5),
            0.5 / ((128 * 3 * 3) ** 0.5),
            0.5 / ((256 * 3 * 3) ** 0.5),
            0.5 / ((512 * 3 * 3) ** 0.5),
            0.5 / ((512 * 2 * 2) ** 0.5),
        ]

        interactions = [
            InteractionConvMaxPool2d(3,   128, 3, h_out=16, w_out=16, padding=1,
                                     weight_gain=weight_gains[0], bias_gain=bias_gains[0]),
            InteractionConvMaxPool2d(128, 256, 3, h_out=8,  w_out=8,  padding=1,
                                     weight_gain=weight_gains[1], bias_gain=bias_gains[1]),
            InteractionConvMaxPool2d(256, 512, 3, h_out=4,  w_out=4,  padding=1,
                                     weight_gain=weight_gains[2], bias_gain=bias_gains[2]),
            InteractionConvMaxPool2d(512, 512, 3, h_out=2,  w_out=2,  padding=1,
                                     weight_gain=weight_gains[3], bias_gain=bias_gains[3]),
            InteractionLinear(512*2*2, 10,
                              weight_gain=weight_gains[4], bias_gain=bias_gains[4]),
        ]

        state_shapes = [
            (128, 16, 16),
            (256, 8, 8),
            (512, 4, 4),
            (512, 2, 2),
            (10,)
        ]

        activations = [hard_sigmoid] * 4 + [identity] 

        connections = [
            (interactions[0], (0, 1)),
            (interactions[1], (1, 2)),
            (interactions[2], (2, 3)),
            (interactions[3], (3, 4)),
            (interactions[4], (4, 5))
        ]

        super().__init__(state_shapes, interactions, connections, activations)
        self.cost_type = 'CE'


class ResNet13_Interactions(InteractionBaseHopfieldModel):
    def __init__(self, num_inputs=3, num_outputs=10):
        nn.Module.__init__(self)

        weight_gains = [
            0.6, 0.6, 0.7,
            0.6, 0.7, 0.6,
            0.6, 0.7, 0.6,
            0.6, 0.7, 0.6,
            0.8
        ]
        bias_gains = [0.5 / np.sqrt(ni * 3 * 3) for ni in
                     [num_inputs, 128, num_inputs, 128, 256, 128, 256, 512, 256, 512, 512, 512, 512]]

        interactions = [
            InteractionConv2d(3, 128, 3, h_out=32, w_out=32, stride=1, padding=1,
                              weight_gain=weight_gains[0], bias_gain=bias_gains[0]),
            InteractionConv2d(128, 128, 3, h_out=16, w_out=16, stride=2, padding=1,
                              weight_gain=weight_gains[0], bias_gain=bias_gains[1]),
            InteractionConv2d(3, 128, 1, h_out=16, w_out=16, stride=2, padding=0,
                              weight_gain=weight_gains[0], bias_gain=bias_gains[2]),
            InteractionConv2d(128, 256, 3, h_out=16, w_out=16, stride=1, padding=1,
                              weight_gain=weight_gains[1], bias_gain=bias_gains[3]),
            InteractionConv2d(256, 256, 3, h_out=8, w_out=8, stride=2, padding=1,
                              weight_gain=weight_gains[1], bias_gain=bias_gains[4]),
            InteractionConv2d(128, 256, 1, h_out=8, w_out=8, stride=2, padding=0,
                              weight_gain=weight_gains[1], bias_gain=bias_gains[5]),
            InteractionConv2d(256, 512, 3, h_out=8, w_out=8, stride=1, padding=1,
                              weight_gain=weight_gains[2], bias_gain=bias_gains[6]),
            InteractionConv2d(512, 512, 3, h_out=4, w_out=4, stride=2, padding=1,
                              weight_gain=weight_gains[2], bias_gain=bias_gains[7]),
            InteractionConv2d(256, 512, 1, h_out=4, w_out=4, stride=2, padding=0,
                              weight_gain=weight_gains[2], bias_gain=bias_gains[8]),
            InteractionConv2d(512, 512, 3, h_out=4, w_out=4, stride=1, padding=1,
                              weight_gain=weight_gains[3], bias_gain=bias_gains[9]),
            InteractionConv2d(512, 512, 3, h_out=2, w_out=2, stride=2, padding=1,
                              weight_gain=weight_gains[3], bias_gain=bias_gains[10]),
            InteractionConv2d(512, 512, 1, h_out=2, w_out=2, stride=2, padding=0,
                              weight_gain=weight_gains[3], bias_gain=bias_gains[11]),
            InteractionLinear(512*2*2, num_outputs,
                              weight_gain=weight_gains[4], bias_gain=bias_gains[12])
        ]

        state_shapes = [
            (128, 32, 32),
            (128, 16, 16),
            (256, 16, 16),
            (256, 8, 8),
            (512, 8, 8),
            (512, 4, 4),
            (512, 4, 4),
            (512, 2, 2),
            (num_outputs,)
        ]

        activations = [hard_sigmoid] * 8 + [identity]

        connections = [
            (interactions[0], (0, 1)),
            (interactions[1], (1, 2)),
            (interactions[2], (0, 2)),
            (interactions[3], (2, 3)),
            (interactions[4], (3, 4)),
            (interactions[5], (2, 4)),
            (interactions[6], (4, 5)),
            (interactions[7], (5, 6)),
            (interactions[8], (4, 6)),
            (interactions[9], (6, 7)),
            (interactions[10], (7, 8)),
            (interactions[11], (6, 8)),
            (interactions[12], (8, 9))
        ]

        super().__init__(state_shapes, interactions, connections, activations)
        self.cost_type = 'CE'


class ResNet16_Interactions(InteractionBaseHopfieldModel):
    def __init__(self, num_inputs=3, num_outputs=10):
        nn.Module.__init__(self)
        
        weight_gains = [
            0.6, 0.6, 0.7,
            0.6, 0.7, 0.6,
            0.6, 0.7, 0.6,
            0.6, 0.7, 0.6,
            0.6, 0.7, 0.6,
            0.8
        ]
        bias_gains = [
            0.5 / np.sqrt(num_inputs * 3 * 3),
            0.5 / np.sqrt(128 * 3 * 3),
            0.5 / np.sqrt(num_inputs * 1 * 1),
            0.5 / np.sqrt(128 * 3 * 3),
            0.5 / np.sqrt(256 * 3 * 3),
            0.5 / np.sqrt(128 * 1 * 1),
            0.5 / np.sqrt(256 * 3 * 3),
            0.5 / np.sqrt(512 * 3 * 3),
            0.5 / np.sqrt(256 * 1 * 1),
            0.5 / np.sqrt(512 * 3 * 3),
            0.5 / np.sqrt(512 * 3 * 3),
            0.5 / np.sqrt(512 * 1 * 1),
            0.5 / np.sqrt(512 * 3 * 3),
            0.5 / np.sqrt(1024 * 3 * 3),
            0.5 / np.sqrt(512 * 1 * 1),
            0.5 / np.sqrt(1024 * 2 * 2),
        ]

        interactions = [
            # Block 1
            InteractionConv2d(3, 128, 3, h_out=32, w_out=32, stride=1, padding=1,
                         weight_gain=weight_gains[0], bias_gain=bias_gains[0]),
            InteractionConv2d(128, 128, 3, h_out=16, w_out=16, stride=2, padding=1,
                         weight_gain=weight_gains[0], bias_gain=bias_gains[1]),
            InteractionConv2d(3, 128, 1, h_out=16, w_out=16, stride=2, padding=0,
                         weight_gain=weight_gains[0], bias_gain=bias_gains[2]),
            # Block 2
            InteractionConv2d(128, 256, 3, h_out=16, w_out=16, stride=1, padding=1,
                         weight_gain=weight_gains[1], bias_gain=bias_gains[3]),
            InteractionConv2d(256, 256, 3, h_out=8, w_out=8, stride=2, padding=1,
                         weight_gain=weight_gains[1], bias_gain=bias_gains[4]),
            InteractionConv2d(128, 256, 1, h_out=8, w_out=8, stride=2, padding=0,
                         weight_gain=weight_gains[1], bias_gain=bias_gains[5]),
            # Block 3
            InteractionConv2d(256, 512, 3, h_out=8, w_out=8, stride=1, padding=1,
                         weight_gain=weight_gains[2], bias_gain=bias_gains[6]),
            InteractionConv2d(512, 512, 3, h_out=4, w_out=4, stride=2, padding=1,
                         weight_gain=weight_gains[2], bias_gain=bias_gains[7]),
            InteractionConv2d(256, 512, 1, h_out=4, w_out=4, stride=2, padding=0,
                         weight_gain=weight_gains[2], bias_gain=bias_gains[8]),
            # Block 4
            InteractionConv2d(512, 512, 3, h_out=4, w_out=4, stride=1, padding=1,
                         weight_gain=weight_gains[3], bias_gain=bias_gains[9]),
            InteractionConv2d(512, 512, 3, h_out=2, w_out=2, stride=2, padding=1,
                         weight_gain=weight_gains[3], bias_gain=bias_gains[10]),
            InteractionConv2d(512, 512, 1, h_out=2, w_out=2, stride=2, padding=0,
                         weight_gain=weight_gains[3], bias_gain=bias_gains[11]),
            # Block 5
            InteractionConv2d(512, 1024, 3, h_out=2, w_out=2, stride=1, padding=1,
                         weight_gain=weight_gains[4], bias_gain=bias_gains[12]),
            InteractionConv2d(1024, 1024, 3, h_out=2, w_out=2, stride=1, padding=1,
                         weight_gain=weight_gains[4], bias_gain=bias_gains[13]),
            InteractionConv2d(512, 1024, 1, h_out=2, w_out=2, stride=1, padding=0,
                         weight_gain=weight_gains[4], bias_gain=bias_gains[14]),
            # Final linear
            InteractionLinear(1024*2*2, num_outputs,
                         weight_gain=weight_gains[5], bias_gain=bias_gains[15])
        ]

        state_shapes = [
            (128, 32, 32),  # s1
            (128, 16, 16),  # s2
            (256, 16, 16),  # s3
            (256, 8, 8),    # s4
            (512, 8, 8),    # s5
            (512, 4, 4),    # s6
            (512, 4, 4),    # s7
            (512, 2, 2),    # s8
            (1024, 2, 2),   # s9
            (1024, 2, 2),   # s10
            (num_outputs,)  # s11 (logits)
        ]
        
        activations = [hard_sigmoid] * 10 + [identity]

        connections = [
            # Block 1
            (interactions[0], (0, 1)),  # conv1
            (interactions[1], (1, 2)),  # conv2
            (interactions[2], (0, 2)),  # skip
            # Block 2
            (interactions[3], (2, 3)),
            (interactions[4], (3, 4)),
            (interactions[5], (2, 4)),
            # Block 3
            (interactions[6], (4, 5)),
            (interactions[7], (5, 6)),
            (interactions[8], (4, 6)),
            # Block 4
            (interactions[9], (6, 7)),
            (interactions[10], (7, 8)),
            (interactions[11], (6, 8)),
            # Block 5
            (interactions[12], (8, 9)),
            (interactions[13], (9, 10)),
            (interactions[14], (8, 10)),
            # Final dense
            (interactions[15], (10, 11))
        ]

        super().__init__(state_shapes, interactions, connections, activations)
        self.cost_type = 'CE'



def train_batch_centered(model, x, y, optimizer, beta=0.1, use_mean_reduction=True,
                         n_iters_free=50, n_iters_nudged=50, streams=None, previous_states=None):
    B = x.size(0)
    device = x.device

    if previous_states is not None and previous_states[0].size(0) == B:
        states = previous_states
    else:
        states = model.create_states(B, device)

    free_states = model.minimize(x, states, beta=0.0, n_iters=n_iters_free, streams=streams)
    free_states = [s.detach() for s in free_states]
    logits_free = free_states[-1]

    # Centered nudging
    b1, b2, denom = compute_betas('centered', beta)

    # Phase 1
    nudged_states = model.minimize(x, free_states, beta=b1, target=y, n_iters=n_iters_nudged, streams=streams)
    nudged_states = [s.detach() for s in nudged_states]

    E_1 = model.energy(x, nudged_states, b1, target=y)
    E_1 = E_1.mean() if use_mean_reduction else E_1.sum()
    grads_1 = torch.autograd.grad(E_1, model.parameters(), create_graph=False)

    # Phase 2 (restart from free)
    nudged_states = model.minimize(x, free_states, beta=b2, target=y, n_iters=n_iters_nudged, streams=streams)
    nudged_states = [s.detach() for s in nudged_states]
    E_2 = model.energy(x, nudged_states, b2, target=y)
    E_2 = E_2.mean() if use_mean_reduction else E_2.sum()
    grads_2 = torch.autograd.grad(E_2, model.parameters(), create_graph=False)

    # EP update
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
        x = x.to(device, memory_format=torch.channels_last)
        y = y.to(device)
        B = x.size(0)
        states = model.create_states(B, x.device)
        free_states = model.minimize(x, states, beta=0.0, n_iters=n_iters_infer, streams=streams)
        logits = free_states[-1]
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += B
    return correct / max(1, total)
