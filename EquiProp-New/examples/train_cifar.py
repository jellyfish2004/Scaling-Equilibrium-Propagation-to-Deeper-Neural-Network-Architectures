import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Optional
from tqdm import tqdm
import time

from eqprop.functional import compute_betas as _compute_betas
from eqprop.core import train_batch_centered, evaluate, ConvHopfieldEnergy32, ResNet13, ResNet16
from dataset import _build_cifar10_loaders

torch.set_float32_matmul_precision('high')

def _sgd_optimizer(model: nn.Module, lr_weights: float = 3e-2, momentum: float = 0.9, weight_decay: float = 2.5e-4):
    # CHANGE: simple SGD with momentum/weight_decay similar to original
    return torch.optim.SGD(model.parameters(), lr=lr_weights, momentum=momentum, weight_decay=weight_decay, nesterov=True)


def _train_loop():
    # CHANGE: runnable entry point to train on CIFAR-10 similar to the original main
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--iters_infer', type=int, default=120)   # free phase iterations
    parser.add_argument('--iters_train', type=int, default=50)    # nudged phase iterations
    parser.add_argument('--cost', type=str, default='MSE', choices=['CE','MSE'])
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--no_normalize', action='store_true')
    parser.add_argument('--checkpoint_every', type=int, default=5)
    parser.add_argument('--cosine_annealing', action='store_true')
    parser.add_argument('--parallel_minimize', action='store_true', help='Enable multi-stream minimize (parallel layer backward, experimental)')
    parser.add_argument('--model', type=str, default='resnet_16')
    parser.add_argument('--compile', action='store_true')
    args = parser.parse_args()

    device = args.device
    # CHANGE: enable cuDNN autotune for conv speedups
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = _build_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        normalize=not args.no_normalize,
    )

    if args.model not in ['resnet_16', 'resnet_13', 'vgg5']:
        raise ValueError(f"Unsupported model '{args.model}'. Choose from ['resnet_16', 'resnet_13', 'vgg5'].")
    if args.model == 'resnet_16':
        model = ResNet16()
    elif args.model == 'resnet_13':
        model = ResNet13()
    elif args.model == 'vgg5':
        model = ConvHopfieldEnergy32()
    model.cost_type = args.cost
    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    if args.compile:
        for layer in model.layers:
            layer.energy = torch.compile(layer.energy, options={
                "epilogue_fusion": True,      # fuse pointwise ops into GEMM templates
                "shape_padding": True,        # pad shapes for better Tensor Core alignment
                # "triton.cudagraphs": True,    # cudagraphs
                "max_autotune": True,         # ensure autotuning is active
                "max_autotune_gemm_backends": "triton",  # broaden autotune search
                "benchmark_kernel": True,     # profile kernel variants to pick fastest
            })
            layer.backward = torch.compile(layer.backward, options={
                "epilogue_fusion": True,      # fuse pointwise ops into GEMM templates
                "shape_padding": True,        # pad shapes for better Tensor Core alignment
                # "triton.cudagraphs": True,    # cudagraphs
                "max_autotune": True,         # ensure autotuning is active
                "max_autotune_gemm_backends": "triton",  # broaden autotune search
                "benchmark_kernel": True,     # profile kernel variants to pick fastest
            })

    optimizer = _sgd_optimizer(model)

    scheduler=None
    if args.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = args.epochs,
            eta_min = 2e-6,
        )
        sched_cfg = dict(
            type   = "CosineAnnealingLR",
            T_max  = args.epochs,
            eta_min= 2e-6,
        )

    print(f"Device: {device}")
    print(f"Cost: {model.cost_type}")
    print(f"Epochs: {args.epochs}, batch_size={args.batch_size}")

    # Set up CUDA streams for parallel minimize if requested
    if args.parallel_minimize:
        streams = [torch.cuda.Stream() for _ in range(len(model.layers))]
        print("[Info] Parallel minimize enabled: using multi-stream per-layer backward.")
    else:
        streams = None
    # All minimize, train, and evaluate calls receive streams now

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_E_1 = 0.0
        running_E_2 = 0.0
        running_correct = 0
        running_count = 0
        iterator = train_loader
        use_tqdm = (tqdm is not None) and (not args.no_tqdm)
        if use_tqdm:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        previous_states = None
        for step, (x, y) in enumerate(iterator):
            x = x.to(device, memory_format=torch.channels_last)
            y = y.to(device)
            E_1, E_2, logits_free, batch_loss, previous_states = train_batch_centered(
                model, x, y, optimizer,
                beta=args.beta, use_mean_reduction=True,
                n_iters_free=args.iters_infer, n_iters_nudged=args.iters_train,
                streams=streams,
                previous_states=previous_states,
                grad_clip=1.0,
            )
            if step==0:
                print(f"first batch (torch compile): {time.time() - start_time}")
            batch_size = x.size(0)
            pred = logits_free.argmax(dim=1)
            correct = (pred == y).sum().item()
            running_correct += correct
            running_count += batch_size
            running_loss += batch_loss
            running_E_1 += float(E_1)
            running_E_2 += float(E_2)
            if use_tqdm and (step % 10 == 0):
                iterator.set_postfix({
                    'E_1': f"{E_1:.4f}",
                    'E_nudged': f"{E_2:.4f}",
                    'acc': f"{(correct / max(1, batch_size)) * 100:.2f}%",
                    'loss': f"{batch_loss:.4f}",
                })
        acc = evaluate(model, test_loader, device=device, n_iters_infer=args.iters_infer, streams=streams)
        denom = max(1, len(train_loader))
        avg_loss = running_loss / denom
        avg_E_1 = running_E_1 / denom
        avg_E_2 = running_E_2 / denom
        train_acc = running_correct / max(1, running_count)
        print(f"Epoch {epoch:03d} | E_free={avg_E_1:.4f} | E_nudged={avg_E_2:.4f} | loss={avg_loss:.4f} | train_acc={train_acc*100:.2f}% | test_acc={acc*100:.2f}%")
        if scheduler:
            scheduler.step()

        if epoch % args.checkpoint_every == 0:
            torch.save(model.state_dict(), f'./checkpoints/cifar_epoch_{epoch}.pt')


if __name__ == '__main__':
    _train_loop()

