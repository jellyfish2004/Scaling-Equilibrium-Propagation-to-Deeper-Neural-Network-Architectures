"""
Debug script for original EquiProp implementation.
Dumps states and gradients during training for comparison with new implementation.

Run from repo root:
    python tests/debug_dump_original.py --batches 2 --dump_every 1

Code references are provided for each section to enable verification against original files.
"""

import argparse
import os
import sys
import torch
import numpy as np

# Setup paths relative to this file's location (tests/)
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
equiprop_orig_dir = os.path.join(root_dir, "EquiProp")
sys.path.insert(0, equiprop_orig_dir)

# === Imports (from dhcnresnet.py lines 1-20) ===
from model.hopfield.network import ConvHopfieldEnergy32
from model.function.network import Network
from model.function.cost import SquaredError
from model.hopfield.minimizer import FixedPointMinimizer
from training.sgd import EquilibriumProp, AugmentedFunction
from training.monitor import Optimizer


def save_model_weights(energy_fn, path):
    """Save model weights (biases + weights) to a file."""
    params = energy_fn.params()
    weights_dict = {}
    for i, p in enumerate(params):
        weights_dict[f'param_{i}'] = p.get().detach().cpu().clone()
    torch.save(weights_dict, path)


def save_states(layers, path):
    """Save layer states to a file (excludes input layer)."""
    states_dict = {}
    # Skip layer 0 (input layer) - only save hidden states
    for i, layer in enumerate(layers[1:]):
        states_dict[f'state_{i}'] = layer.state.detach().cpu().clone()
    torch.save(states_dict, path)


def save_gradients(grads, path):
    """Save gradients to a file."""
    grads_dict = {}
    for i, g in enumerate(grads):
        grads_dict[f'grad_{i}'] = g.detach().cpu().clone()
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


def main():
    parser = argparse.ArgumentParser(description='Debug dump script for original EquiProp')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--dump_every', type=int, default=50, help='Dump every N batches')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dump_dir', type=str, default='dumps/original', help='Directory to save dumps')
    parser.add_argument('--beta', type=float, default=0.1, help='Nudging factor')
    parser.add_argument('--num_iterations_inference', type=int, default=120, help='Free phase iterations')
    parser.add_argument('--num_iterations_training', type=int, default=50, help='Nudged phase iterations')
    parser.add_argument('--save_initial_weights', type=str, default='dumps/initial_weights.pt', 
                        help='Path to save initial weights (used by new script)')
    parser.add_argument('--save_inputs', action='store_true',
                        help='Save inputs for later reuse by new implementation')
    parser.add_argument('--load_inputs', action='store_true',
                        help='Load inputs from saved file instead of dataloader')
    parser.add_argument('--inputs_file', type=str, default='dumps/shared_inputs.pt',
                        help='Path for shared inputs file')
    parser.add_argument('--batches', type=int, default=None,
                        help='Limit number of batches to process')
    parser.add_argument('--mode', type=str, default='asynchronous',
                        choices=['synchronous', 'asynchronous', 'forward', 'backward'],
                        help='Minimizer update mode: synchronous, asynchronous (default), forward, backward')
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
    os.makedirs(os.path.dirname(args.save_initial_weights), exist_ok=True)

    # === Data loading (deterministic - no RandAugment for reproducibility) ===
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
    
    training_data = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=train_transform)
    
    # Use a seeded generator for reproducible shuffle order
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    training_loader = torch.utils.data.DataLoader(
        training_data, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0,  # 0 for full determinism
        pin_memory=True,
        generator=g
    )

    # Setup device and dtype
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64 if args.float64 else torch.float32
    print(f"Running on {device} with dtype={dtype}")

    # === Model setup (from model/hopfield/network.py: ConvHopfieldEnergy32.__init__) ===
    weight_gains = [0.4, 0.7, 0.6, 0.3, 0.6]  # 5 layers
    activation = 'hard-sigmoid'
    num_inputs = 3
    num_outputs = 10

    energy_fn = ConvHopfieldEnergy32(num_inputs, num_outputs, weight_gains=weight_gains, activation=activation)
    energy_fn.set_device(device)
    
    # Convert to float64 if requested
    if args.float64:
        for param in energy_fn.params():
            param.state = param.state.double()
        for layer in energy_fn.layers():
            if hasattr(layer, 'state') and layer.state is not None:
                layer.state = layer.state.double()
        print("Converted model to float64")

    # Save initial weights (before any training)
    save_model_weights(energy_fn, args.save_initial_weights)
    print(f"Saved initial weights to {args.save_initial_weights}")

    # === Network wrapper ===
    network = Network(energy_fn)

    # === Cost function ===
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredError(output_layer)

    # === Augmented function ===
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)

    # === Minimizers ===
    params = energy_fn.params()
    layers = energy_fn.layers()
    free_layers = network.free_layers()

    minimizer_inference = FixedPointMinimizer(energy_fn, free_layers)
    minimizer_inference.num_iterations = args.num_iterations_inference

    minimizer_training = FixedPointMinimizer(augmented_fn, free_layers)
    minimizer_training.num_iterations = args.num_iterations_training

    # Set the update mode for both minimizers
    minimizer_inference.mode = args.mode
    minimizer_training.mode = args.mode

    # === Gradient estimator ===
    estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, minimizer_training)
    estimator.variant = 'centered'
    estimator.nudging = args.beta

    # === Optimizer ===
    learning_rates_weights = [3e-2] * len(weight_gains)
    learning_rates_biases = learning_rates_weights.copy()
    learning_rates = learning_rates_biases + learning_rates_weights
    momentum = 0.9
    weight_decay = 2.5e-4
    optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

    print(f"Training for {args.epochs} epochs, dumping every {args.dump_every} batches")
    print(f"Dump directory: {args.dump_dir}")
    print(f"Update mode: {args.mode}")

    # Prepare shared inputs if loading
    shared_inputs = None
    if args.load_inputs:
        shared_inputs = torch.load(args.inputs_file)
        print(f"Loaded shared inputs from {args.inputs_file} ({len(shared_inputs)} batches)")
    
    # List to collect inputs for saving
    inputs_to_save = []

    # === Training loop ===
    global_batch_idx = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch} ===")
        
        for batch_idx, (x, y) in enumerate(training_loader):
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
            
            # Collect inputs for saving
            if args.save_inputs:
                inputs_to_save.append({'x': x.detach().cpu().clone(), 'y': y.detach().cpu().clone()})
            
            should_dump = (global_batch_idx % args.dump_every == 0)
            dump_prefix = os.path.join(args.dump_dir, f'batch_{global_batch_idx:05d}')

            # Save input if dumping
            if should_dump:
                save_input_batch(x, y, f'{dump_prefix}_input.pt')
                save_model_weights(energy_fn, f'{dump_prefix}_weights_before.pt')

            # === Free Phase ===
            network.set_input(x, reset=False)
            minimizer_inference.compute_equilibrium()
            
            free_states = [layer.state.detach().clone() for layer in layers]
            
            if should_dump:
                save_states(layers, f'{dump_prefix}_free_states.pt')
                with torch.no_grad():
                    energy_free = energy_fn.eval().mean().item()

            # === Set target ===
            cost_fn.set_target(y)

            # === Positive Nudged Phase (+beta) ===
            for layer, state in zip(layers, free_states):
                layer.state = state.clone()
            
            augmented_fn.nudging = args.beta
            minimizer_training.compute_equilibrium()
            
            pos_nudged_states = [layer.state.detach().clone() for layer in layers]
            
            if should_dump:
                save_states(layers, f'{dump_prefix}_pos_nudged_states.pt')
                with torch.no_grad():
                    energy_pos = energy_fn.eval().mean().item()

            # === Negative Nudged Phase (-beta) ===
            for layer, state in zip(layers, free_states):
                layer.state = state.clone()
            
            augmented_fn.nudging = -args.beta
            minimizer_training.compute_equilibrium()
            
            neg_nudged_states = [layer.state.detach().clone() for layer in layers]
            
            if should_dump:
                save_states(layers, f'{dump_prefix}_neg_nudged_states.pt')
                with torch.no_grad():
                    energy_neg = energy_fn.eval().mean().item()
                
                save_energies(energy_free, energy_pos, energy_neg, f'{dump_prefix}_energies.pt')

            # === Compute Gradients ===
            for layer, state in zip(layers, free_states):
                layer.state = state.clone()
            
            grads = estimator.compute_gradient()
            
            if should_dump:
                save_gradients(grads, f'{dump_prefix}_gradients.pt')
                print(f"  Dumped batch {global_batch_idx}: E_free={energy_free:.4f}, E_pos={energy_pos:.4f}, E_neg={energy_neg:.4f}")

            # === Update Parameters ===
            all_params = params + cost_fn.params()
            for param, grad in zip(all_params, grads):
                param.state.grad = grad
            
            optimizer.step()
            
            for param in all_params:
                param.clamp_()

            if should_dump:
                save_model_weights(energy_fn, f'{dump_prefix}_weights_after.pt')

            global_batch_idx += 1

            # Progress logging
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}/{len(training_loader)}")

    print(f"\nTraining complete. Dumps saved to {args.dump_dir}")
    print(f"Initial weights saved to {args.save_initial_weights}")
    
    # Save shared inputs if requested
    if args.save_inputs and inputs_to_save:
        torch.save(inputs_to_save, args.inputs_file)
        print(f"Saved {len(inputs_to_save)} input batches to {args.inputs_file}")


if __name__ == '__main__':
    main()
