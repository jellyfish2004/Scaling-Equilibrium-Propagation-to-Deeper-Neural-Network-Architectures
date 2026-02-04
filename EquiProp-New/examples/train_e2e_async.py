#!/usr/bin/env python3
"""
End-to-End Training Script with Selectable Update Mode.

This script demonstrates training with different layer update modes:
- synchronous: all layers updated at once (Jacobi style)
- asynchronous: even layers first, then odd layers (Gauss-Seidel style, more stable)
- forward: layers updated one at a time from first to last
- backward: layers updated one at a time from last to first

Usage:
    python train_e2e_async.py --mode asynchronous --epochs 10
    python train_e2e_async.py --mode synchronous --epochs 10 --model vgg5
"""

import sys
import os
import time
import torch
import torch.nn.functional as F
from typing import List
from tqdm import tqdm
import argparse

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from eqprop.interactions.core import (
    train_batch_centered, evaluate,
    ConvHopfieldEnergy32_Interactions,
    ResNet13_Interactions,
    ResNet16_Interactions
)
from eqprop.activation import relu6, hard_sigmoid
from dataset import _build_cifar10_loaders


def main():
    parser = argparse.ArgumentParser(description='E2E Training with Selectable Update Mode')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--beta', type=float, default=0.1, help='Nudging factor')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--iters_infer', type=int, default=120, help='Free phase iterations')
    parser.add_argument('--iters_train', type=int, default=50, help='Nudged phase iterations')
    parser.add_argument('--cost', type=str, default='MSE', choices=['CE', 'MSE'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--model', type=str, default='vgg5', choices=['vgg5', 'resnet_13', 'resnet_16'])
    parser.add_argument('--activation', type=str, default='hard_sigmoid', choices=['relu6', 'hard_sigmoid'])
    parser.add_argument('--lr', type=float, default=3e-2, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight_decay', type=float, default=2.5e-4, help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--mode', type=str, default='asynchronous',
                        choices=['synchronous', 'asynchronous', 'forward', 'backward'],
                        help='Layer update mode (default: asynchronous)')
    parser.add_argument('--batches', type=int, default=None,
                        help='Limit number of batches per epoch (for quick testing)')
    parser.add_argument('--no_wandb', action='store_true', default=True, help='Disable wandb logging')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    
    device = args.device
    torch.backends.cudnn.benchmark = True

    # Load data
    train_loader, test_loader = _build_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        normalize=True,
    )

    # Select activation
    act_fn = relu6 if args.activation == 'relu6' else hard_sigmoid

    # Build model
    if args.model == 'resnet_16':
        model = ResNet16_Interactions()
    elif args.model == 'resnet_13':
        model = ResNet13_Interactions(activation=act_fn)
    else:  # vgg5
        model = ConvHopfieldEnergy32_Interactions(activation=act_fn)

    model.cost_type = args.cost
    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    print("=" * 60)
    print(f"E2E Training with Update Mode: {args.mode}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Cost: {model.cost_type}")
    print(f"Activation: {args.activation}")
    print(f"Update Mode: {args.mode}")
    print(f"Epochs: {args.epochs}, batch_size={args.batch_size}")
    print(f"Iterations: free={args.iters_infer}, nudged={args.iters_train}")
    print("=" * 60)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_E_free = 0.0
        running_correct = 0
        running_count = 0

        iterator = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        previous_states = None
        
        for step, (x, y) in enumerate(iterator):
            # Apply batch limit if specified
            if args.batches is not None and step >= args.batches:
                break

            x = x.to(device, memory_format=torch.channels_last)
            y = y.to(device)

            E_free, E_1, E_2, logits_free, batch_loss, previous_states = train_batch_centered(
                model, x, y, optimizer,
                beta=args.beta,
                use_mean_reduction=True,
                n_iters_free=args.iters_infer,
                n_iters_nudged=args.iters_train,
                previous_states=previous_states,
                grad_clip=args.grad_clip,
                mode=args.mode  # Use the specified update mode
            )

            batch_size = x.size(0)
            pred = logits_free.argmax(dim=1)
            correct = (pred == y).sum().item()

            running_correct += correct
            running_count += batch_size
            running_loss += batch_loss
            running_E_free += float(E_free)

            iterator.set_postfix({
                'E_free': f"{E_free:.4f}",
                'acc': f"{(correct / max(1, batch_size)) * 100:.2f}%",
                'loss': f"{batch_loss:.4f}",
            })

        # Compute epoch stats
        denom = step + 1 if args.batches is None else min(step + 1, args.batches)
        avg_loss = running_loss / max(1, denom)
        avg_E_free = running_E_free / max(1, denom)
        train_acc = running_correct / max(1, running_count)

        # Evaluate on test set
        acc = evaluate(model, test_loader, device=device, n_iters_infer=args.iters_infer)

        print(f"Epoch {epoch:03d} | E_free={avg_E_free:.4f} | loss={avg_loss:.4f} | "
              f"train_acc={train_acc*100:.2f}% | test_acc={acc*100:.2f}%")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final test accuracy: {acc*100:.2f}%")


if __name__ == '__main__':
    main()
