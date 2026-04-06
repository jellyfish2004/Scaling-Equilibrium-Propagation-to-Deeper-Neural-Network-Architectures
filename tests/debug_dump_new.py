"""
Debug script for new EquiProp interactions implementation.
Dumps states and gradients during training for comparison with original implementation.

Run from repo root:
    python tests/debug_dump_new.py --batches 2 --dump_every 1

This script:
1. Loads the original model and creates the new model
2. Syncs weights from original to new
3. Trains using the same hyperparameters
4. Dumps states/gradients at same intervals for comparison
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

# Setup paths relative to this file's location (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

# Add both EquiProp-New and EquiProp to path
eqprop_new_dir = os.path.join(root_dir, "EquiProp-New")
equiprop_orig_dir = os.path.join(root_dir, "EquiProp")
sys.path.insert(0, eqprop_new_dir)
sys.path.insert(0, equiprop_orig_dir)

from eqprop.core import ConvHopfieldEnergy32_Interactions
from eqprop.activation import hard_sigmoid
from eqprop.functional import compute_betas


def sync_weights_vgg5(original_model, new_model):
    """Sync weights from original VGG5 model to new interactions model.
    
    Original param order: biases[0-4], weights[5-9]
      - Bias shapes: (128,), (256,), (512,), (512,), (10,)
      - Weight shapes: (128,3,3,3), (256,128,3,3), (512,256,3,3), (512,512,3,3), (512,2,2,10)
    
    New param order: alternating (bias, weight) for 5 interactions
      - Bias shapes: (128,1,1), (256,1,1), (512,1,1), (512,1,1), (10,)
      - Weight shapes: (128,3,3,3), (256,128,3,3), (512,256,3,3), (512,512,3,3), (10,2048)
    """
    orig_params = original_model.params()  # List: biases[0:5], weights[5:10]
    new_params = list(new_model.parameters())  # List: [bias0, weight0, bias1, weight1, ...]
    
    with torch.no_grad():
        for i in range(5):  # 5 layers
            orig_bias = orig_params[i].get()        # bias i (1D)
            orig_weight = orig_params[5 + i].get()  # weight i
            
            # New model: bias at 2*i, weight at 2*i+1
            new_bias = new_params[2 * i]
            new_weight = new_params[2 * i + 1]
            
            # Copy bias (reshape if needed: 1D -> 3D for conv layers)
            if i < 4:  # Conv layers have 3D bias (C, 1, 1)
                new_bias.copy_(orig_bias.view(-1, 1, 1))
            else:  # Linear layer has 1D bias
                new_bias.copy_(orig_bias)
            
            # Copy weight (last layer needs reshape)
            if i == 4:
                # Original: (512, 2, 2, 10) -> New: (10, 2048)
                new_weight.copy_(orig_weight.permute(3, 0, 1, 2).reshape(10, -1))
            else:
                new_weight.copy_(orig_weight)


def load_original_model(device='cuda'):
    """Load the original VGG5 model for weight syncing."""
    import model.hopfield.network as orig_net
    
    weight_gains = [0.4, 0.7, 0.6, 0.3, 0.6]  # 5 layers
    
    energy_fn = orig_net.ConvHopfieldEnergy32(3, 10, weight_gains=weight_gains, activation='hard-sigmoid')
    energy_fn.set_device(device)
    return energy_fn


def save_model_weights(model, path):
    """Save model weights to a file."""
    weights_dict = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        weights_dict[name] = param.detach().cpu().clone()
    torch.save(weights_dict, path)


def save_states(states, path):
    """Save states to a file."""
    states_dict = {}
    for i, state in enumerate(states):
        states_dict[f'state_{i}'] = state.detach().cpu().clone()
    torch.save(states_dict, path)


def save_gradients(grads, model, path):
    """Save gradients to a file."""
    grads_dict = {}
    for i, (name, g) in enumerate(zip([n for n, _ in model.named_parameters()], grads)):
        grads_dict[name] = g.detach().cpu().clone()
    torch.save(grads_dict, path)


def save_energies(energy_free, energy_pos, energy_neg, path):
    """Save energy values to a file."""
    torch.save({
        'energy_free': energy_free,
        'energy_pos': energy_pos,
        'energy_neg': energy_neg,
    }, path)


def save_input_batch(x, y, path):
    """Save input batch for reproducibility."""
    torch.save({
        'x': x.detach().cpu().clone(),
        'y': y.detach().cpu().clone(),
    }, path)


def minimize_with_dump(model, x, states, beta=0.0, target=None, n_iters=20, dump_prefix=None, max_dump_iters=10):
    """
    Minimize energy while dumping intermediate states (for debugging).
    Only dumps first max_dump_iters iterations to save memory.
    
    Returns:
        final_states: List of equilibrated state tensors
        iteration_states: List of (iter_num, states_dict) for dumped iterations
    """
    current_states = list(states)
    iteration_states = []
    num_layers = len(current_states)
    
    for iter_num in range(n_iters):
        # Dump first few iterations to trace divergence
        if iter_num < max_dump_iters:
            states_dict = {}
            for i, s in enumerate(current_states):
                states_dict[f'layer_{i}'] = s.detach().cpu().clone()
            iteration_states.append((iter_num, states_dict))
        
        # Single minimize step (synchronous)
        current_states = model.minimize_step(x, current_states, beta, target)
    
    return current_states, iteration_states


def train_batch_with_dump(model, x, y, optimizer, beta=0.1,
                          n_iters_free=120, n_iters_nudged=50,
                          previous_states=None, grad_clip=1.0, manual_grads=False,
                          mode='asynchronous'):
    """
    Train one batch using centered EP and return intermediate states for dumping.
    
    Args:
        model: The EquiProp model
        x, y: Input batch and labels
        optimizer: PyTorch optimizer
        beta: Nudging coefficient
        n_iters_free, n_iters_nudged: Number of minimization iterations
        previous_states: Optional previous states for warm start
        grad_clip: Gradient clipping value
        manual_grads: If True, use manual gradient computation (matches original)
    
    Returns:
        dict with all values needed for dumping
    """
    B = x.size(0)
    device = x.device

    if previous_states is not None and previous_states[0].size(0) == B:
        states = previous_states
    else:
        states = model.create_states(B, device)

    # === Free Phase ===
    free_states = model.minimize(x, states, beta=0.0, n_iters=n_iters_free, mode=mode)
    free_states = [s.detach() for s in free_states]
    logits_free = free_states[-1]

    # Compute energy at free equilibrium
    with torch.no_grad():
        E_free = model.energy(x, free_states, beta=0.0).mean().item()

    # Centered nudging betas
    # Note: compute_betas returns (b1=-beta, b2=+beta, denom=2*beta)
    # But original debug script uses +beta for "pos" and -beta for "neg"
    # So we swap: pos uses +beta, neg uses -beta for consistent comparison
    b1, b2, denom = compute_betas('centered', beta)
    # b1=-beta (used in original as second/neg phase), b2=+beta (used in original as first/pos phase)

    # === Positive Nudged Phase (beta=+beta, matching original) ===
    pos_nudged_states = model.minimize(x, free_states, beta=+beta, target=y, n_iters=n_iters_nudged, mode=mode)
    pos_nudged_states = [s.detach() for s in pos_nudged_states]

    E_pos_tensor = model.energy(x, pos_nudged_states)  # Hopfield energy only
    E_pos = E_pos_tensor.mean().item()

    # === Negative Nudged Phase (beta=-beta, matching original) ===
    neg_nudged_states = model.minimize(x, free_states, beta=-beta, target=y, n_iters=n_iters_nudged, mode=mode)
    neg_nudged_states = [s.detach() for s in neg_nudged_states]
    
    E_neg_tensor = model.energy(x, neg_nudged_states)
    E_neg = E_neg_tensor.mean().item()

    # For gradient computation: use manual gradients or autograd
    if manual_grads:
        # Use manual gradient computation (matches original implementation)
        grads = model.compute_manual_gradients(x, pos_nudged_states, neg_nudged_states, denom)
    else:
        # Use autograd (default)
        grads_pos = torch.autograd.grad(E_pos_tensor.mean(), model.parameters(), create_graph=False)
        grads_neg = torch.autograd.grad(E_neg_tensor.mean(), model.parameters(), create_graph=False)
        # Centered EP: (grads(+beta) - grads(-beta)) / (2*beta)
        grads = [((gp - gn).detach() / denom) for gp, gn in zip(grads_pos, grads_neg)]
    
    # Apply gradient clipping
    if grad_clip is not None:
        total_norm = torch.norm(torch.stack([g.norm() for g in grads]))
        if total_norm > grad_clip:
            scale = grad_clip / (total_norm + 1e-6)
            grads = [g * scale for g in grads]
            
    # === Update Parameters ===
    optimizer.zero_grad()
    for p, g in zip(model.parameters(), grads):
        p.grad = g
    optimizer.step()

    # Task loss from free-phase logits (MSE)
    if model.cost_type == 'CE':
        batch_loss = F.cross_entropy(logits_free, y).item()
    else:
        one_hot = F.one_hot(y, num_classes=logits_free.shape[1]).float()
        batch_loss = (0.5 * ((logits_free - one_hot) ** 2).sum(dim=1)).mean().item()

    return {
        'free_states': free_states,
        'pos_nudged_states': pos_nudged_states,
        'neg_nudged_states': neg_nudged_states,
        'grads': grads,
        'E_free': E_free,
        'E_pos': E_pos,
        'E_neg': E_neg,
        'batch_loss': batch_loss,
        'logits_free': logits_free,
    }


def main():
    parser = argparse.ArgumentParser(description='Debug dump script for new EquiProp interactions')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dump_every', type=int, default=50, help='Dump every N batches')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dump_dir', type=str, default='dumps/new', help='Directory to save dumps')
    parser.add_argument('--beta', type=float, default=0.1, help='Nudging factor')
    parser.add_argument('--num_iterations_inference', type=int, default=120, help='Free phase iterations')
    parser.add_argument('--num_iterations_training', type=int, default=50, help='Nudged phase iterations')
    parser.add_argument('--load_initial_weights', type=str, default='dumps/initial_weights.pt',
                        help='Path to load initial weights from original model')
    parser.add_argument('--load_inputs', action='store_true',
                        help='Load inputs from saved file instead of dataloader')
    parser.add_argument('--inputs_file', type=str, default='dumps/shared_inputs.pt',
                        help='Path for shared inputs file')
    parser.add_argument('--batches', type=int, default=None,
                        help='Limit number of batches to process')
    parser.add_argument('--manual_grads', action='store_true',
                        help='Use manual gradient computation instead of autograd (matches original)')
    parser.add_argument('--mode', type=str, default='asynchronous',
                        choices=['synchronous', 'asynchronous', 'forward', 'backward'],
                        help='Layer update mode: synchronous, asynchronous (default), forward, backward')
    parser.add_argument('--float64', action='store_true',
                        help='Use float64 precision for better numerical accuracy')
    args = parser.parse_args()

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Disable TF32 for reproducibility
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Set float64 if requested (must be before model creation)
    if args.float64:
        torch.set_default_dtype(torch.float64)

    # Create dump directory
    os.makedirs(args.dump_dir, exist_ok=True)

    # Setup device and dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64 if args.float64 else torch.float32
    print(f"Running on {device} with dtype={dtype}")

    # Load data with deterministic ordering (matching original)
    import torchvision
    import torchvision.transforms as T
    
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023*3, 0.1994*3, 0.2010*3)
    
    # Disable ALL random augmentations for exact reproducibility between runs
    train_transform = T.Compose([
        # T.RandomHorizontalFlip(0.5),  # Disabled for reproducibility
        # T.RandomCrop(size=[32,32], padding=4, padding_mode='edge'),  # Disabled for reproducibility
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    
    train_set = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    
    # Use a seeded generator for reproducible shuffle order
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # 0 for full determinism
        pin_memory=True,
        generator=g
    )

    # === Setup Models ===
    print("Loading original model for weight sync...")
    original_model = load_original_model(device)
    
    print("Creating new model...")
    new_model = ConvHopfieldEnergy32_Interactions(activation=hard_sigmoid)
    new_model.cost_type = 'MSE'  # Match original SquaredError cost
    new_model.to(device)
    
    # Convert model to float64 if requested
    if args.float64:
        new_model = new_model.double()
        print("Converted model to float64")

    # Sync weights from original to new
    print("Syncing weights from original to new model...")
    sync_weights_vgg5(original_model, new_model)

    # Save initial weights (should match original)
    os.makedirs(os.path.dirname(args.load_initial_weights), exist_ok=True)
    save_model_weights(new_model, os.path.join(args.dump_dir, 'initial_weights.pt'))
    print(f"Saved synced initial weights")

    # Optimizer (SGD with momentum and weight decay)
    lr = 3e-2
    momentum = 0.9
    weight_decay = 2.5e-4
    optimizer = torch.optim.SGD(
        new_model.parameters(), 
        lr=lr, 
        momentum=momentum, 
        weight_decay=weight_decay, 
        nesterov=True
    )

    print(f"Training for {args.epochs} epochs, dumping every {args.dump_every} batches")
    print(f"Dump directory: {args.dump_dir}")
    print(f"Update mode: {args.mode}")

    # Prepare shared inputs if loading
    shared_inputs = None
    if args.load_inputs:
        shared_inputs = torch.load(args.inputs_file)
        print(f"Loaded shared inputs from {args.inputs_file} ({len(shared_inputs)} batches)")

    # Training loop
    global_batch_idx = 0
    previous_states = None
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        new_model.train()
        
        for batch_idx, (x, y) in enumerate(train_loader):
            # Apply batch limit if specified
            if args.batches is not None and global_batch_idx >= args.batches:
                break
            
            # Load from shared inputs if specified
            if args.load_inputs and shared_inputs is not None:
                if global_batch_idx >= len(shared_inputs):
                    print(f"Ran out of shared inputs at batch {global_batch_idx}")
                    break
                x = shared_inputs[global_batch_idx]['x'].to(device=device, dtype=dtype)
                y = shared_inputs[global_batch_idx]['y'].to(device)
            else:
                x = x.to(device=device, dtype=dtype)
                y = y.to(device)
            
            should_dump = (global_batch_idx % args.dump_every == 0)
            dump_prefix = os.path.join(args.dump_dir, f'batch_{global_batch_idx:05d}')

            # Save input if dumping
            if should_dump:
                save_input_batch(x, y, f'{dump_prefix}_input.pt')
                save_model_weights(new_model, f'{dump_prefix}_weights_before.pt')

            # Train one batch and get intermediate values
            result = train_batch_with_dump(
                new_model, x, y, optimizer,
                beta=args.beta,
                n_iters_free=args.num_iterations_inference,
                n_iters_nudged=args.num_iterations_training,
                previous_states=previous_states,
                grad_clip=1.0,
                manual_grads=args.manual_grads,
                mode=args.mode
            )
            
            previous_states = result['pos_nudged_states']  # Match original: 2nd phase uses +beta

            if should_dump:
                save_states(result['free_states'], f'{dump_prefix}_free_states.pt')
                save_states(result['pos_nudged_states'], f'{dump_prefix}_pos_nudged_states.pt')
                save_states(result['neg_nudged_states'], f'{dump_prefix}_neg_nudged_states.pt')
                save_gradients(result['grads'], new_model, f'{dump_prefix}_gradients.pt')
                save_energies(result['E_free'], result['E_pos'], result['E_neg'], f'{dump_prefix}_energies.pt')
                save_model_weights(new_model, f'{dump_prefix}_weights_after.pt')
                
                print(f"  Dumped batch {global_batch_idx}: E_free={result['E_free']:.4f}, "
                      f"E_pos={result['E_pos']:.4f}, E_neg={result['E_neg']:.4f}")

            global_batch_idx += 1

            # Progress logging
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}")

    print(f"\nTraining complete. Dumps saved to {args.dump_dir}")


if __name__ == '__main__':
    main()
