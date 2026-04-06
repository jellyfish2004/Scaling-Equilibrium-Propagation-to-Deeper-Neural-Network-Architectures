#!/usr/bin/env python3
"""
Timing comparison: Original EquiProp vs New EquiProp implementation.

Compares the same two implementations tested in run_comparison.sh,
but focused on speed without deterministic/reproducibility overhead.

Run from repo root:
    python tests/timing_comparison.py
    python tests/timing_comparison.py --batches 20 --warmup 5
"""

import argparse
import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
eqprop_new_dir = os.path.join(root_dir, "EquiProp-New")
equiprop_orig_dir = os.path.join(root_dir, "EquiProp")
sys.path.insert(0, eqprop_new_dir)
sys.path.insert(0, equiprop_orig_dir)

# --- New implementation imports ---
from eqprop.core import (
    ConvHopfieldEnergy32_Interactions,
    train_batch_centered,
)
from eqprop.activation import hard_sigmoid

# --- Original implementation imports ---
from model.hopfield.network import ConvHopfieldEnergy32
from model.function.network import Network
from model.function.cost import SquaredError
from model.hopfield.minimizer import FixedPointMinimizer
from training.sgd import EquilibriumProp, AugmentedFunction
from training.monitor import Optimizer

torch.set_float32_matmul_precision('high')

def setup_original_model(device, mode='asynchronous'):
    """Setup the original EquiProp model with all its wrappers."""
    weight_gains = [0.4, 0.7, 0.6, 0.3, 0.6]
    energy_fn = ConvHopfieldEnergy32(3, 10, weight_gains=weight_gains, activation='hard-sigmoid')
    energy_fn.set_device(device)

    network = Network(energy_fn)
    output_layer = energy_fn.layers()[-1]
    cost_fn = SquaredError(output_layer)
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)

    params = energy_fn.params()
    layers = energy_fn.layers()
    free_layers = network.free_layers()

    minimizer_inference = FixedPointMinimizer(energy_fn, free_layers)
    minimizer_inference.num_iterations = 120
    minimizer_inference.mode = mode

    minimizer_training = FixedPointMinimizer(augmented_fn, free_layers)
    minimizer_training.num_iterations = 50
    minimizer_training.mode = mode

    estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, minimizer_training)
    estimator.variant = 'centered'
    estimator.nudging = 0.1

    learning_rates = [3e-2] * 10
    optimizer = Optimizer(energy_fn, cost_fn, learning_rates, 0.9, 2.5e-4)

    return {
        'energy_fn': energy_fn,
        'network': network,
        'cost_fn': cost_fn,
        'augmented_fn': augmented_fn,
        'minimizer_inference': minimizer_inference,
        'minimizer_training': minimizer_training,
        'estimator': estimator,
        'optimizer': optimizer,
        'params': params,
        'layers': layers,
    }


def train_batch_original(ctx, x, y, beta=0.1):
    """Train one batch with the original implementation."""
    network = ctx['network']
    energy_fn = ctx['energy_fn']
    cost_fn = ctx['cost_fn']
    augmented_fn = ctx['augmented_fn']
    minimizer_inference = ctx['minimizer_inference']
    estimator = ctx['estimator']
    optimizer = ctx['optimizer']
    params = ctx['params']
    layers = ctx['layers']

    # Free phase
    network.set_input(x, reset=False)
    minimizer_inference.compute_equilibrium()
    free_states = [layer.state.detach().clone() for layer in layers]

    # Set target
    cost_fn.set_target(y)

    # Positive nudged phase
    for layer, state in zip(layers, free_states):
        layer.state = state.clone()
    augmented_fn.nudging = beta
    ctx['minimizer_training'].compute_equilibrium()

    # Negative nudged phase
    for layer, state in zip(layers, free_states):
        layer.state = state.clone()
    augmented_fn.nudging = -beta
    ctx['minimizer_training'].compute_equilibrium()

    # Compute gradients
    for layer, state in zip(layers, free_states):
        layer.state = state.clone()
    grads = estimator.compute_gradient()

    # Update
    all_params = params + cost_fn.params()
    for param, grad in zip(all_params, grads):
        param.state.grad = grad
    optimizer.step()
    for param in all_params:
        param.clamp_()


def setup_new_model(device):
    """Setup the new EquiProp interactions model."""
    model = ConvHopfieldEnergy32_Interactions(activation=hard_sigmoid)
    model.cost_type = 'MSE'
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=3e-2,
        momentum=0.9,
        weight_decay=2.5e-4,
        nesterov=True,
    )
    return model, optimizer


def benchmark(fn, warmup, repeats, device):
    """Benchmark a function with CUDA timing."""
    use_cuda = device == 'cuda'

    # Warmup
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    times = []
    if use_cuda:
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            start_ev.record()
            fn()
            end_ev.record()
            torch.cuda.synchronize()
            times.append(start_ev.elapsed_time(end_ev))
    else:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
    return times


def main():
    parser = argparse.ArgumentParser(description='Timing: Original vs New EquiProp')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--warmup', type=int, default=3, help='Warmup repetitions')
    parser.add_argument('--repeats', type=int, default=10, help='Timed repetitions')
    parser.add_argument('--mode', type=str, default='asynchronous',
                        choices=['synchronous', 'asynchronous'])
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--iters_free', type=int, default=120)
    parser.add_argument('--iters_nudged', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cudnn.benchmark = True  # let cuDNN auto-tune

    print("=" * 60)
    print("  Original vs New EquiProp — Timing")
    print("=" * 60)
    print(f"  Device:      {device}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Mode:        {args.mode}")
    print(f"  Iters:       free={args.iters_free}, nudged={args.iters_nudged}")
    print(f"  Warmup:      {args.warmup}")
    print(f"  Repeats:     {args.repeats}")
    print("=" * 60)

    # --- Generate shared input data ---
    x = torch.randn(args.batch_size, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (args.batch_size,), device=device)

    # ============================================================
    # Original implementation
    # ============================================================
    print("\nSetting up ORIGINAL model...")
    orig_ctx = setup_original_model(device, mode=args.mode)
    orig_ctx['minimizer_inference'].num_iterations = args.iters_free
    orig_ctx['minimizer_training'].num_iterations = args.iters_nudged

    def run_original():
        train_batch_original(orig_ctx, x.clone(), y.clone(), beta=args.beta)

    print("Benchmarking ORIGINAL...")
    times_orig = benchmark(run_original, args.warmup, args.repeats, device)

    # ============================================================
    # New implementation
    # ============================================================
    print("\nSetting up NEW model...")
    new_model, new_optimizer = setup_new_model(device)
    previous_states = [None]  # mutable container for state persistence

    def run_new():
        _, _, _, _, _, prev, _ = train_batch_centered(
            new_model, x.clone(), y.clone(), new_optimizer,
            beta=args.beta,
            n_iters_free=args.iters_free,
            n_iters_nudged=args.iters_nudged,
            previous_states=previous_states[0],
            grad_clip=1.0,
            mode=args.mode,
        )
        previous_states[0] = prev

    print("Benchmarking NEW...")
    times_new = benchmark(run_new, args.warmup, args.repeats, device)

    # ============================================================
    # New implementation + torch.compile
    # ============================================================
    print("\nSetting up NEW + torch.compile model...")
    compiled_model, compiled_optimizer = setup_new_model(device)
    
    # Pre-compute async plans before compiling (so they're baked in as constants)
    compiled_model._precompute_async_plan()
    
    # Compile the actual hotpath methods
    compiled_model.minimize = torch.compile(
        compiled_model.minimize,
        options={"epilogue_fusion": True, "max_autotune": True, "triton.cudagraphs": True}
    )
    compiled_model.energy = torch.compile(
        compiled_model.energy,
        options={"epilogue_fusion": True, "max_autotune": True}
    )
    
    compiled_prev_states = [None]

    def run_compiled():
        _, _, _, _, _, prev, _ = train_batch_centered(
            compiled_model, x.clone(), y.clone(), compiled_optimizer,
            beta=args.beta,
            n_iters_free=args.iters_free,
            n_iters_nudged=args.iters_nudged,
            previous_states=compiled_prev_states[0],
            grad_clip=1.0,
            mode=args.mode,
        )
        compiled_prev_states[0] = prev

    # Extra warmup for torch.compile (first calls trigger compilation)
    print("Benchmarking NEW + torch.compile (extra warmup for compilation)...")
    times_compiled = benchmark(run_compiled, args.warmup + 5, args.repeats, device)

    # ============================================================
    # Results
    # ============================================================
    orig_mean = np.mean(times_orig)
    orig_std = np.std(times_orig)
    new_mean = np.mean(times_new)
    new_std = np.std(times_new)
    compiled_mean = np.mean(times_compiled)
    compiled_std = np.std(times_compiled)

    print(f"\n{'='*60}")
    print("RESULTS (per training batch: free + 2x nudged + grad + update)")
    print(f"{'='*60}")
    print(f"  {'Variant':<25s} {'Mean':>10s} {'Std':>10s} {'vs Orig':>10s}")
    print(f"  {'-'*55}")
    print(f"  {'Original':<25s} {orig_mean:8.2f}ms {orig_std:8.2f}ms {'1.000x':>10s}")
    
    speedup_new = orig_mean / new_mean
    print(f"  {'New':<25s} {new_mean:8.2f}ms {new_std:8.2f}ms {speedup_new:9.3f}x")
    
    speedup_compiled = orig_mean / compiled_mean
    print(f"  {'New + torch.compile':<25s} {compiled_mean:8.2f}ms {compiled_std:8.2f}ms {speedup_compiled:9.3f}x")

    print()


if __name__ == '__main__':
    main()

