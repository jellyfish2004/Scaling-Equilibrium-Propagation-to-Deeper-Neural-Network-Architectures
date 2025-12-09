import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from tqdm import tqdm
import argparse
import wandb

from eqprop.interactions.core import (
    train_batch_centered, evaluate, 
    ConvHopfieldEnergy32_Interactions, 
    ResNet13_Interactions, 
    ResNet16_Interactions
)
from eqprop.activation import relu6, hard_sigmoid
from dataset import _build_cifar10_loaders

torch.set_float32_matmul_precision('high')


# ============================================================================
# Metric computation helpers (matching original EquiProp Monitor/Statistics)
# ============================================================================

def compute_state_norms(states: List[torch.Tensor]) -> Dict[str, float]:
    """Compute mean absolute value (Norm) for each state tensor."""
    norms = {}
    for state_idx, state in enumerate(states):
        key = f"Norm/state_{state_idx}"
        norms[key] = torch.abs(state).mean().item()
    return norms


def compute_state_saturations(states: List[torch.Tensor]) -> Dict[str, float]:
    """Compute saturation (fraction at 0) for each state tensor."""
    saturations = {}
    for state_idx, state in enumerate(states):
        key = f"Saturation/state_{state_idx}"
        saturations[key] = (state == 0.0).float().mean().item() * 100  # percentage
    return saturations


def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """Compute mean absolute gradient for each parameter."""
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            clean_name = name.replace('.', '_')
            grad_stats[f"Gradient/{clean_name}"] = torch.abs(param.grad).mean().item()
    return grad_stats


def evaluate_with_stats(model, dataloader, device: str, n_iters_infer: int = 120, streams=None):
    """Evaluate model and compute full statistics matching original EquiProp."""
    model.eval()
    total = 0
    correct = 0
    top5_correct = 0
    running_energy = 0.0
    running_cost = 0.0
    
    # Accumulators for state stats
    norm_accumulators = {}
    saturation_accumulators = {}
    
    for x, y in tqdm(dataloader, desc="Evaluating"):
        x = x.to(device, memory_format=torch.channels_last)
        y = y.to(device)
        B = x.size(0)
        states = model.create_states(B, device)
        free_states = model.minimize(x, states, beta=0.0, n_iters=n_iters_infer, streams=streams)
        logits = free_states[-1]
        
        # Accuracy
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        
        # Top-5 accuracy
        _, top5_pred = logits.topk(min(5, logits.size(1)), dim=1)
        top5_correct += top5_pred.eq(y.view(-1, 1).expand_as(top5_pred)).any(dim=1).sum().item()
        
        # Energy (at free equilibrium)
        with torch.no_grad():
            energy = model.energy(x, free_states, beta=0.0).mean().item()
            running_energy += energy
        
        # Cost
        if model.cost_type == 'CE':
            cost = F.cross_entropy(logits, y).item()
        else:
            one_hot = F.one_hot(y, num_classes=logits.shape[1]).float()
            cost = (0.5 * ((logits - one_hot) ** 2).sum(dim=1)).mean().item()
        running_cost += cost
        
        # State norms and saturations
        batch_norms = compute_state_norms(free_states)
        batch_saturations = compute_state_saturations(free_states)
        
        for k, v in batch_norms.items():
            norm_accumulators[k] = norm_accumulators.get(k, 0.0) + v
        for k, v in batch_saturations.items():
            saturation_accumulators[k] = saturation_accumulators.get(k, 0.0) + v
        
        total += B
    
    n_batches = len(dataloader)
    stats = {
        "Error/test": (1.0 - correct / max(1, total)) * 100,
        "Top5Error/test": (1.0 - top5_correct / max(1, total)) * 100,
        "Energy/test": running_energy / max(1, n_batches),
        "Cost/test": running_cost / max(1, n_batches),
    }
    
    # Average state stats
    for k, v in norm_accumulators.items():
        stats[k.replace("Norm/", "Norm/test_")] = v / max(1, n_batches)
    for k, v in saturation_accumulators.items():
        stats[k.replace("Saturation/", "Saturation/test_")] = v / max(1, n_batches)
    
    return correct / max(1, total), stats


def _sgd_optimizer(model: nn.Module, lr_weights: float = 3e-2, momentum: float = 0.9, weight_decay: float = 2.5e-4):
    return torch.optim.SGD(model.parameters(), lr=lr_weights, momentum=momentum, weight_decay=weight_decay, nesterov=True)


def _train_loop():
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
    parser.add_argument('--no_wandb', action='store_true', default=False, help="If set, do not use wandb to log results")
    parser.add_argument('--wandb_project', type=str, default='EqProp-New-Interactions', help="Wandb project name")
    parser.add_argument('--lr', type=float, default=3e-2, help="Learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum")
    parser.add_argument('--weight_decay', type=float, default=2.5e-4, help="Weight decay")
    parser.add_argument('--activation', type=str, default='relu6', choices=['relu6', 'hard_sigmoid'], help="Activation function")
    args = parser.parse_args()

    device = args.device
    torch.backends.cudnn.benchmark = True

    train_loader, test_loader = _build_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        normalize=not args.no_normalize,
    )

    if args.model not in ['resnet_16', 'resnet_13', 'vgg5']:
        raise ValueError(f"Unsupported model '{args.model}'. Choose from ['resnet_16', 'resnet_13', 'vgg5'].")
    
    # Select activation function
    if args.activation == 'relu6':
        act_fn = relu6
    else:
        act_fn = hard_sigmoid
    
    if args.model == 'resnet_16':
        model = ResNet16_Interactions(activation=act_fn)
    elif args.model == 'resnet_13':
        model = ResNet13_Interactions(activation=act_fn)
    elif args.model == 'vgg5':
        model = ConvHopfieldEnergy32_Interactions(activation=act_fn)
    
    model.cost_type = args.cost
    model.to(device)
    model = model.to(memory_format=torch.channels_last)

    if args.compile:
        # Compile top-level methods for better graph optimization
        model.energy = torch.compile(model.energy, options={
            "epilogue_fusion": True,
            "max_autotune": True,
        })
        model.minimize_step = torch.compile(model.minimize_step, options={
            "epilogue_fusion": True,
            "max_autotune": True,
            "triton.cudagraphs": True,
        })

    optimizer = _sgd_optimizer(model, lr_weights=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Initialize wandb logging
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model}_beta{args.beta}_lr{args.lr:.0e}",
            config={
                "dataset": "CIFAR10",
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "beta": args.beta,
                "num_iterations_inference": args.iters_infer,
                "num_iterations_training": args.iters_train,
                "learning_rate": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "cost": args.cost,
                "device": device,
                "compile": args.compile,
                "cosine_annealing": args.cosine_annealing,
            }
        )

    scheduler=None
    if args.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = args.epochs,
            eta_min = 2e-6,
        )

    print(f"Device: {device}")
    print(f"Cost: {model.cost_type}")
    print(f"Epochs: {args.epochs}, batch_size={args.batch_size}")

    if args.parallel_minimize:
        streams = [torch.cuda.Stream() for _ in range(len(model.connections))]
        print("[Info] Parallel minimize enabled: using multi-stream per-interaction backward.")
    else:
        streams = None

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_E_free = 0.0  # Energy at free equilibrium
        running_E_1 = 0.0     # Energy at positive nudged equilibrium
        running_E_2 = 0.0     # Energy at negative nudged equilibrium
        running_correct = 0
        running_top5_correct = 0
        running_count = 0
        
        # Accumulators for state stats (training)
        train_norm_accumulators = {}
        train_saturation_accumulators = {}
        train_gradient_accumulators = {}
        gradient_count = 0
        
        iterator = train_loader
        use_tqdm = (tqdm is not None) and (not args.no_tqdm)
        if use_tqdm:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        previous_states = None
        for step, (x, y) in enumerate(iterator):
            x = x.to(device, memory_format=torch.channels_last)
            y = y.to(device)
            E_free, E_1, E_2, logits_free, batch_loss, previous_states = train_batch_centered(
                model, x, y, optimizer,
                beta=args.beta, use_mean_reduction=True,
                n_iters_free=args.iters_infer, n_iters_nudged=args.iters_train,
                streams=streams,
                previous_states=previous_states,
                grad_clip=1.0
            )
            if step==0:
                print(f"first batch (torch compile): {time.time() - start_time}")
            batch_size = x.size(0)
            pred = logits_free.argmax(dim=1)
            correct = (pred == y).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = logits_free.topk(min(5, logits_free.size(1)), dim=1)
            top5_correct = top5_pred.eq(y.view(-1, 1).expand_as(top5_pred)).any(dim=1).sum().item()
            running_top5_correct += top5_correct
            
            running_correct += correct
            running_count += batch_size
            running_loss += batch_loss
            running_E_free += float(E_free)
            running_E_1 += float(E_1)
            running_E_2 += float(E_2)
            
            # Accumulate state norms and saturations from free_states
            if wandb.run is not None:
                batch_norms = compute_state_norms(previous_states)
                batch_saturations = compute_state_saturations(previous_states)
                for k, v in batch_norms.items():
                    train_norm_accumulators[k] = train_norm_accumulators.get(k, 0.0) + v
                for k, v in batch_saturations.items():
                    train_saturation_accumulators[k] = train_saturation_accumulators.get(k, 0.0) + v
                
                # Accumulate gradient stats (gradients are set after optimizer step)
                batch_grads = compute_gradient_stats(model)
                for k, v in batch_grads.items():
                    train_gradient_accumulators[k] = train_gradient_accumulators.get(k, 0.0) + v
                gradient_count += 1
            
            if use_tqdm and (step % 1 == 0):
                iterator.set_postfix({
                    'E_free': f"{E_free:.4f}",
                    'E_nudged': f"{E_1:.4f}",
                    'acc': f"{(correct / max(1, batch_size)) * 100:.2f}%",
                    'loss': f"{batch_loss:.4f}",
                })
        
        # Evaluation with full stats
        if wandb.run is not None:
            acc, test_stats = evaluate_with_stats(model, test_loader, device=device, 
                                                   n_iters_infer=args.iters_infer, streams=streams)
        else:
            acc = evaluate(model, test_loader, device=device, n_iters_infer=args.iters_infer, streams=streams)
            test_stats = {}
        
        denom = max(1, len(train_loader))
        avg_loss = running_loss / denom
        avg_E_free = running_E_free / denom
        avg_E_1 = running_E_1 / denom
        avg_E_2 = running_E_2 / denom
        train_acc = running_correct / max(1, running_count)
        train_top5_acc = running_top5_correct / max(1, running_count)
        
        print(f"Epoch {epoch:03d} | E_free={avg_E_free:.4f} | E_nudged={avg_E_1:.4f} | loss={avg_loss:.4f} | train_acc={train_acc*100:.2f}% | test_acc={acc*100:.2f}%")
        
        # Log metrics to wandb
        if wandb.run is not None:
            log_dict = {
                # Core training metrics
                "Energy/inference": avg_E_free,  # Free equilibrium energy (matches original EquiProp)
                "Energy_nudged_pos/train": avg_E_1,  # Positive nudged equilibrium energy
                "Energy_nudged_neg/train": avg_E_2,  # Negative nudged equilibrium energy
                "Cost/train": avg_loss,
                "Error/train": (1.0 - train_acc) * 100,
                "Top5Error/train": (1.0 - train_top5_acc) * 100,
            }
            
            # Add all test stats
            log_dict.update(test_stats)
            
            # Average state norms and saturations for training
            for k, v in train_norm_accumulators.items():
                log_dict[k.replace("Norm/", "Norm/train_")] = v / denom
            for k, v in train_saturation_accumulators.items():
                log_dict[k.replace("Saturation/", "Saturation/train_")] = v / denom
            
            # Average gradient stats
            for k, v in train_gradient_accumulators.items():
                log_dict[k] = v / max(1, gradient_count)
            
            if scheduler:
                log_dict["learning_rate"] = scheduler.get_last_lr()[0]
            
            wandb.log(log_dict, step=epoch)
        
        if scheduler:
            scheduler.step()

        if epoch % args.checkpoint_every == 0:
            if not os.path.exists('./checkpoints'):
                os.makedirs('./checkpoints')
            torch.save(model.state_dict(), f'./checkpoints/cifar_interactions_epoch_{epoch}.pt')

if __name__ == '__main__':
    _train_loop()
