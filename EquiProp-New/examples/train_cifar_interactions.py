import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
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


def evaluate_with_stats(model, dataloader, device: str, n_iters_infer: int = 120, streams=None, mode='asynchronous'):
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
        x = x.to(device)
        y = y.to(device)
        B = x.size(0)
        states = model.create_states(B, device)
        free_states = model.minimize(x, states, beta=0.0, n_iters=n_iters_infer, mode=mode, streams=streams)
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


# ============================================================================
# Checkpoint utilities
# ============================================================================

def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, wandb_run_id: Optional[str] = None):
    """Save a complete checkpoint for resuming training."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'wandb_run_id': wandb_run_id,
    }
    torch.save(checkpoint, path)
    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(path: str, model, optimizer, scheduler=None, device='cuda'):
    """Load a checkpoint and return the epoch to resume from."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    wandb_run_id = checkpoint.get('wandb_run_id', None)
    print(f"[Checkpoint] Loaded from {path}, resuming after epoch {epoch}")
    return epoch, wandb_run_id


def _train_loop():
    parser = argparse.ArgumentParser()
    # Training config
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--iters_infer', type=int, default=120, help='Free phase iterations')
    parser.add_argument('--iters_train', type=int, default=50, help='Nudged phase iterations')
    parser.add_argument('--cost', type=str, default='MSE', choices=['CE','MSE'])
    parser.add_argument('--workers', type=int, default=16)
    
    # Model config
    parser.add_argument('--model', type=str, default='resnet_16', choices=['resnet_16', 'resnet_13', 'vgg5'])
    parser.add_argument('--activation', type=str, default='relu6', choices=['relu6', 'hard_sigmoid'])
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for optimization')
    
    # Optimizer config
    parser.add_argument('--lr', type=float, default=3e-2, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=2.5e-4)
    parser.add_argument('--cosine_annealing', action='store_true')
    parser.add_argument('--parallel_minimize', action='store_true', help='Enable multi-stream minimize')
    parser.add_argument('--mode', type=str, default='asynchronous', choices=['synchronous', 'asynchronous'],
                        help='Minimize mode: synchronous (all-at-once) or asynchronous (alternating even/odd)')
    
    # Logging config
    parser.add_argument('--no_tqdm', action='store_true')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='EqProp-New-Interactions')
    parser.add_argument('--log_batch_every', type=int, default=0, help='Log batch metrics every N steps (0=disabled)')
    
    # Checkpoint config
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--checkpoint_every', type=int, default=5, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()

    device = args.device
    torch.backends.cudnn.benchmark = True

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data loading
    train_loader, test_loader = _build_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        normalize=True,
    )

    # Model setup
    act_fn = relu6 if args.activation == 'relu6' else hard_sigmoid
    
    if args.model == 'resnet_16':
        model = ResNet16_Interactions(activation=act_fn)
    elif args.model == 'resnet_13':
        model = ResNet13_Interactions(activation=act_fn)
    elif args.model == 'vgg5':
        model = ConvHopfieldEnergy32_Interactions(activation=act_fn)
    
    model.cost_type = args.cost
    model.to(device)
    # Note: channels_last removed for parity with original EquiProp

    if args.compile:
        model.energy = torch.compile(model.energy, options={"epilogue_fusion": True, "max_autotune": True})
        model.minimize_step = torch.compile(model.minimize_step, options={"epilogue_fusion": True, "max_autotune": True, "triton.cudagraphs": True})

    # Optimizer and scheduler
    optimizer = _sgd_optimizer(model, lr_weights=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = None
    if args.cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-6)

    # Resume from checkpoint if specified
    start_epoch = 1
    wandb_run_id = None
    if args.resume:
        start_epoch, wandb_run_id = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch += 1  # Resume from next epoch

    # Initialize wandb
    if not args.no_wandb:
        wandb_kwargs = {
            'project': args.wandb_project,
            'name': f"{args.model}_beta{args.beta}_lr{args.lr:.0e}",
            'config': {
                "dataset": "CIFAR10",
                "model": args.model,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "beta": args.beta,
                "iters_infer": args.iters_infer,
                "iters_train": args.iters_train,
                "lr": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
                "cost": args.cost,
                "activation": args.activation,
                "cosine_annealing": args.cosine_annealing,
            }
        }
        if wandb_run_id:
            wandb_kwargs['id'] = wandb_run_id
            wandb_kwargs['resume'] = 'allow'
        wandb.init(**wandb_kwargs)

    # Parallel streams for multi-GPU
    streams = None
    if args.parallel_minimize and torch.cuda.is_available():
        streams = [torch.cuda.Stream() for _ in range(len(model.connections))]
        print("[Info] Parallel minimize enabled")

    print(f"Device: {device}")
    print(f"Model: {args.model}, Cost: {model.cost_type}, Activation: {args.activation}")
    print(f"Epochs: {args.epochs}, Starting from: {start_epoch}, Batch size: {args.batch_size}")

    # Global step counter for batch logging
    global_step = (start_epoch - 1) * len(train_loader)

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_E_free = 0.0
        running_E_nudged = 0.0
        running_correct = 0
        running_top5_correct = 0
        running_count = 0
        
        # Accumulators for state/gradient stats
        train_norm_accumulators = {}
        train_saturation_accumulators = {}
        train_gradient_accumulators = {}
        gradient_count = 0
        
        iterator = train_loader
        if not args.no_tqdm:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch:03d}")
        
        previous_states = None
        for step, (x, y) in enumerate(iterator):
            x = x.to(device)
            y = y.to(device)
            
            # Train batch with new return signature
            E_free, E_nudged, logits_free, batch_loss, previous_states, free_states, layerwise_energies = train_batch_centered(
                model, x, y, optimizer,
                beta=args.beta, use_mean_reduction=True,
                n_iters_free=args.iters_infer, n_iters_nudged=args.iters_train,
                streams=streams,
                previous_states=previous_states,
                grad_clip=1.0,
                mode=args.mode
            )
            
            if step == 0 and epoch == start_epoch:
                print(f"First batch time: {time.time() - start_time:.2f}s")
            
            batch_size = x.size(0)
            pred = logits_free.argmax(dim=1)
            correct = (pred == y).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = logits_free.topk(min(5, logits_free.size(1)), dim=1)
            top5_correct = top5_pred.eq(y.view(-1, 1).expand_as(top5_pred)).any(dim=1).sum().item()
            
            running_correct += correct
            running_top5_correct += top5_correct
            running_count += batch_size
            running_loss += batch_loss
            running_E_free += E_free
            running_E_nudged += E_nudged
            
            # Per-batch logging
            if args.log_batch_every > 0 and step % args.log_batch_every == 0 and wandb.run:
                batch_log = {
                    "Energy/batch": E_free,
                    "Energy_nudged/batch": E_nudged,
                    "Cost/batch": batch_loss,
                    "Error/batch": (1.0 - correct / batch_size) * 100,
                }
                # Add layerwise energies
                for i, e in enumerate(layerwise_energies):
                    batch_log[f"LayerEnergy/layer_{i}"] = e
                # Add state norms/saturations
                batch_log.update({k: v for k, v in compute_state_norms(free_states).items()})
                batch_log.update({k: v for k, v in compute_state_saturations(free_states).items()})
                # Add gradient stats
                batch_log.update({k: v for k, v in compute_gradient_stats(model).items()})
                wandb.log(batch_log, step=global_step)
            
            # Accumulate stats for epoch-level logging
            if wandb.run:
                for k, v in compute_state_norms(free_states).items():
                    train_norm_accumulators[k] = train_norm_accumulators.get(k, 0.0) + v
                for k, v in compute_state_saturations(free_states).items():
                    train_saturation_accumulators[k] = train_saturation_accumulators.get(k, 0.0) + v
                for k, v in compute_gradient_stats(model).items():
                    train_gradient_accumulators[k] = train_gradient_accumulators.get(k, 0.0) + v
                gradient_count += 1
            
            if not args.no_tqdm:
                iterator.set_postfix({
                    'E_free': f"{E_free:.3f}",
                    'loss': f"{batch_loss:.4f}",
                    'acc': f"{correct/batch_size*100:.1f}%"
                })
            
            global_step += 1
        
        # Evaluation
        if wandb.run:
            acc, test_stats = evaluate_with_stats(model, test_loader, device=device, 
                                                  n_iters_infer=args.iters_infer, streams=streams, mode=args.mode)
        else:
            acc = evaluate(model, test_loader, device=device, n_iters_infer=args.iters_infer, streams=streams)
            test_stats = {}
        
        # Epoch statistics
        n_batches = len(train_loader)
        avg_loss = running_loss / n_batches
        avg_E_free = running_E_free / n_batches
        avg_E_nudged = running_E_nudged / n_batches
        train_acc = running_correct / running_count
        train_top5_acc = running_top5_correct / running_count
        
        print(f"Epoch {epoch:03d} | E_free={avg_E_free:.4f} | E_nudged={avg_E_nudged:.4f} | "
              f"loss={avg_loss:.4f} | train_acc={train_acc*100:.2f}% | test_acc={acc*100:.2f}%")
        
        # Epoch-level wandb logging
        if wandb.run:
            log_dict = {
                "Energy/train": avg_E_free,
                "Energy_nudged/train": avg_E_nudged,
                "Cost/train": avg_loss,
                "Error/train": (1.0 - train_acc) * 100,
                "Top5Error/train": (1.0 - train_top5_acc) * 100,
            }
            log_dict.update(test_stats)
            
            # Average state/gradient stats
            for k, v in train_norm_accumulators.items():
                log_dict[k.replace("state_", "train_state_")] = v / n_batches
            for k, v in train_saturation_accumulators.items():
                log_dict[k.replace("state_", "train_state_")] = v / n_batches
            for k, v in train_gradient_accumulators.items():
                log_dict[k] = v / gradient_count
            
            if scheduler:
                log_dict["LearningRate/train"] = scheduler.get_last_lr()[0]
            
            wandb.log(log_dict, step=global_step)
        
        if scheduler:
            scheduler.step()
        
        # Save checkpoint
        if epoch % args.checkpoint_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'{args.model}_epoch_{epoch}.pt')
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, 
                          wandb.run.id if wandb.run else None)

    print(f"Training complete. Total time: {time.time() - start_time:.1f}s")


if __name__ == '__main__':
    _train_loop()
