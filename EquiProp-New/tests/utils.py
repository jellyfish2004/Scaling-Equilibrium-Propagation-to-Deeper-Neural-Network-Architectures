import torch
import torch.nn as nn
import sys
import os
import numpy as np

def sync_weights_resnet(original_model, new_model):
    """
    Syncs weights for ResNet13 models.
    Original model: SumSeparableFunction with params list (Biases then Weights).
    New model: InteractionBaseHopfieldModel with interactions list (each has weight and bias).
    """
    
    # Original params
    # Biases: 9 items (Layers 1 to 9)
    # Weights: 13 items (Edges)
    
    if callable(original_model.params):
        params = original_model.params()
    else:
        params = original_model.params
    
    # Number of layers with bias = 9 (indices 1 to 9, 0 is input)
    n_biases = 9
    orig_biases = params[:n_biases]
    orig_weights = params[n_biases:]
    
    new_interactions = new_model.interactions
    
    if len(orig_weights) != len(new_interactions):
        print(f"WARNING: Weight count mismatch: {len(orig_weights)} vs {len(new_interactions)}")

    
    weight_map = {i: i for i in range(13)}
    
    with torch.no_grad():
        for orig_idx, new_idx in weight_map.items():
            orig_w = orig_weights[orig_idx].get()
            new_w = new_model.interactions[new_idx].conv.weight if hasattr(new_model.interactions[new_idx], 'conv') else new_model.interactions[new_idx].linear.weight
            
            if orig_w.shape != new_w.shape:
                # Special case for DenseWeight (in, h, w, out) -> Linear (out, in*h*w)
                if orig_w.ndim == 4 and new_w.ndim == 2:
                    # Orig: (C, H, W, Out) or (In_C, In_H, In_W, Out_C)
                    # New: (Out, In_Flat)
                    # We need to permute Orig to (Out, C, H, W) then flatten
                    # Current shape: (512, 2, 2, 10) -> (C, H, W, Out)
                    # Permute to (3, 0, 1, 2) -> (10, 512, 2, 2)
                    print(f"DEBUG: Permuting DenseWeight {orig_w.shape} to match {new_w.shape}")
                    orig_w = orig_w.permute(3, 0, 1, 2).contiguous().view_as(new_w)
                elif orig_w.shape == new_w.t().shape:
                    orig_w = orig_w.t()
                elif orig_w.numel() == new_w.numel():
                     orig_w = orig_w.view_as(new_w)
                else:
                    raise ValueError(f"Cannot broadcast {orig_w.shape} to {new_w.shape}")
            
            new_w.copy_(orig_w)
            if orig_w.shape == new_w.shape:
                 diff = (orig_w - new_w).abs().max()
                 print(f"DEBUG: Synced weight {new_idx}. Max diff: {diff}")
            else:
                 print(f"DEBUG: Synced weight {new_idx} with shape mismatch? Orig: {orig_w.shape}, New: {new_w.shape}")
            
    # Bias mapping

    # Orig Bias 0 -> New Interaction 0
    # Orig Bias 1 -> New Interaction 1
    # Orig Bias 2 -> New Interaction 3
    # Orig Bias 3 -> New Interaction 5
    # Orig Bias 4 -> New Interaction 6
    # Orig Bias 5 -> New Interaction 8
    # Orig Bias 6 -> New Interaction 9
    # Orig Bias 7 -> New Interaction 11
    # Orig Bias 8 -> New Interaction 12
    
    # And set biases of skip interactions (2, 4, 7, 10) to 0.
    
    bias_map = {
        0: 0,
        1: 1,
        2: 3,
        3: 5,
        4: 6,
        5: 8,
        6: 9,
        7: 11,
        8: 12
    }
    
    # Zero out all biases first
    for i in range(13):
        if hasattr(new_model.interactions[i], 'bias'):
            new_model.interactions[i].bias.data.zero_()
            
    with torch.no_grad():
        for orig_idx, new_idx in bias_map.items():
            orig_b = orig_biases[orig_idx].get()
            new_b = new_model.interactions[new_idx].bias
            
            # Handle shape mismatches
            if orig_b.shape != new_b.shape:
                if orig_b.numel() == new_b.numel():
                    orig_b = orig_b.view_as(new_b)
                else:
                    # Try to broadcast?
                    pass
            
            new_b.copy_(orig_b)
                
    print("ResNet weights synchronized successfully.")

def compare_tensors(t1, t2, name, rtol=1e-5, atol=1e-6):
    if t1.shape != t2.shape:
        print(f"[{name}] SHAPE MISMATCH: {t1.shape} vs {t2.shape}")
        return False
    
    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    
    status = "MATCH" if is_close else "FAIL"
    print(f"[{name}] {status} | Max Diff: {max_diff:.2e} | Max Rel. Diff: {(diff/(t1+1e-10)).max():.2e}")
    return is_close


def compare_gradients(orig_grad, new_grad, name, verbose=True):
    """
    Compare two gradient tensors with meaningful metrics.
    
    Returns:
        dict with comparison metrics
    """
    if orig_grad.shape != new_grad.shape:
        print(f"[{name}] SHAPE MISMATCH: {orig_grad.shape} vs {new_grad.shape}")
        return {"match": False, "error": "shape_mismatch"}
    
    # Flatten for easier computation
    o = orig_grad.flatten().float()
    n = new_grad.flatten().float()
    
    # 1. Cosine Similarity (most important - are they pointing same direction?)
    o_norm = torch.norm(o)
    n_norm = torch.norm(n)
    if o_norm > 1e-10 and n_norm > 1e-10:
        cosine_sim = torch.dot(o, n) / (o_norm * n_norm)
        cosine_sim = cosine_sim.item()
    else:
        cosine_sim = float('nan')
    
    # 2. Normalized L2 Error (error relative to gradient magnitude)
    diff = o - n
    l2_error = torch.norm(diff).item()
    normalized_l2 = l2_error / (o_norm.item() + 1e-10)
    
    # 3. Max Absolute Difference
    max_abs_diff = diff.abs().max().item()
    
    # 4. Detailed Relative Error Analysis
    # Use epsilon floor to avoid division by very small values
    eps = 1e-8
    rel_error = diff.abs() / (o.abs() + eps)
    mean_rel_error = rel_error.mean().item()
    max_rel_error = rel_error.max().item()
        
    # 5. Distribution of relative errors (buckets)
    pct_under_1pct = (rel_error < 0.01).float().mean().item() * 100  # < 1%
    pct_under_5pct = (rel_error < 0.05).float().mean().item() * 100  # < 5%
    pct_under_10pct = (rel_error < 0.10).float().mean().item() * 100  # < 10%
    pct_under_50pct = (rel_error < 0.50).float().mean().item() * 100  # < 50%
    pct_over_100pct = (rel_error > 1.0).float().mean().item() * 100  # > 100% (completely off)
    
    # 6. Percentage of elements within tolerance
    atol = 1e-5
    rtol = 1e-3
    close_mask = diff.abs() <= (atol + rtol * o.abs())
    pct_close = close_mask.float().mean().item() * 100
    
    # Determine if match based on cosine similarity and normalized error
    is_match = cosine_sim > 0.9999 and normalized_l2 < 0.01
    
    if verbose:
        status = "✓ MATCH" if is_match else "✗ FAIL"
        print(f"[{name:12s}] {status}")
        print(f"    Cosine Similarity: {cosine_sim:.6f} (1.0 = perfect)")
        print(f"    Normalized L2 Err: {normalized_l2:.2e} (0 = perfect)")
        print(f"    Max Abs Diff:      {max_abs_diff:.2e}")
        print(f"    Mean Rel Error:    {mean_rel_error:.2e}")
        print(f"    Max Rel Error:     {max_rel_error:.2e}")
        print(f"    % Elements Close:  {pct_close:.1f}%")
        print(f"    Rel Error Distribution:")
        print(f"      <1%: {pct_under_1pct:.1f}%  <5%: {pct_under_5pct:.1f}%  <10%: {pct_under_10pct:.1f}%  <50%: {pct_under_50pct:.1f}%  >100%: {pct_over_100pct:.1f}%")
        
    return {
        "match": is_match,
        "cosine_similarity": cosine_sim,
        "normalized_l2_error": normalized_l2,
        "max_abs_diff": max_abs_diff,
        "mean_rel_error": mean_rel_error,
        "max_rel_error": max_rel_error,
        "pct_close": pct_close,
        "pct_under_1pct": pct_under_1pct,
        "pct_under_5pct": pct_under_5pct,
        "pct_under_10pct": pct_under_10pct,
        "pct_under_50pct": pct_under_50pct,
        "pct_over_100pct": pct_over_100pct,
        "orig_norm": o_norm.item(),
        "new_norm": n_norm.item()
    }

