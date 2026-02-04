"""
Compare dumps from original and new EquiProp implementations.

This script:
1. Loads dumps from both implementations
2. Compares states, gradients, energies, and weights
3. Reports differences and identifies where discrepancies occur
"""

import argparse
import os
import glob
import torch
import numpy as np
from collections import defaultdict


def compare_tensors(t1, t2, name, rtol=1e-5, atol=1e-6, verbose=True):
    """Compare two tensors with detailed metrics."""
    if t1.shape != t2.shape:
        if verbose:
            print(f"  [{name}] SHAPE MISMATCH: {t1.shape} vs {t2.shape}")
        return {'match': False, 'error': 'shape_mismatch'}

    print(name)

    diff = (t1 - t2).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    # Cosine similarity
    t1_flat = t1.flatten()
    t2_flat = t2.flatten()
    t1_norm = torch.norm(t1_flat)
    t2_norm = torch.norm(t2_flat)

    
    if t1_norm > 1e-10 and t2_norm > 1e-10:
        cosine_sim = torch.dot(t1_flat, t2_flat) / (t1_norm * t2_norm)
        cosine_sim = cosine_sim.item()
    else:
        cosine_sim = float('nan')
    
    is_close = torch.allclose(t1, t2, rtol=rtol, atol=atol)
    
    result = {
        'match': is_close,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'cosine_sim': cosine_sim,
        't1_norm': t1_norm.item(),
        't2_norm': t2_norm.item(),
    }
    
    if verbose:
        status = "✓ MATCH" if is_close else "✗ DIFF"
        print(f"  [{name}] {status} | Max: {max_diff:.2e} | Mean: {mean_diff:.2e} | Cos: {cosine_sim:.6f}")
    
    return result


def compare_dict_files(file1, file2, name_prefix, verbose=True):
    """Compare two .pt files containing dicts of tensors."""
    if not os.path.exists(file1):
        if verbose:
            print(f"  [{name_prefix}] Original file missing: {file1}")
        return [{'name': name_prefix, 'match': False, 'error': 'missing_original'}]
    
    if not os.path.exists(file2):
        if verbose:
            print(f"  [{name_prefix}] New file missing: {file2}")
        return [{'name': name_prefix, 'match': False, 'error': 'missing_new'}]
    
    dict1 = torch.load(file1, map_location='cpu', weights_only=True)
    dict2 = torch.load(file2, map_location='cpu', weights_only=True)
    
    results = []
    
    # Check if this is a weights/gradients file that needs special handling
    # Original: biases[0-4], weights[5-9] (10 params)
    # New: alternating (bias0, weight0, bias1, weight1, ...) (10 params)
    keys1 = sorted(dict1.keys())
    keys2 = sorted(dict2.keys())
    
    is_param_file = (name_prefix in ['gradients', 'weights_before', 'weights_after'] 
                     and len(keys1) == 10 and len(keys2) == 10)
    
    if is_param_file:
        # Map original order to new order and handle shape differences
        # Original: bias0, bias1, bias2, bias3, bias4, weight0, weight1, weight2, weight3, weight4
        # New: bias0, weight0, bias1, weight1, bias2, weight2, bias3, weight3, bias4, weight4
        for layer_idx in range(5):
            # Bias comparison
            orig_bias_key = keys1[layer_idx]  # biases are first 5
            new_bias_key = keys2[2 * layer_idx]  # bias at even indices
            
            t1 = dict1[orig_bias_key]
            t2 = dict2[new_bias_key]
            
            # Handle shape difference: original (C,) vs new (C,1,1) for conv layers
            if t1.dim() == 1 and t2.dim() == 3:
                t1 = t1.view(-1, 1, 1)
            
            result = compare_tensors(t1, t2, f"{name_prefix}/bias_{layer_idx}", verbose=verbose)
            result['name'] = f"{name_prefix}/bias_{layer_idx}"
            results.append(result)
            
            # Weight comparison
            orig_weight_key = keys1[5 + layer_idx]  # weights are indices 5-9
            new_weight_key = keys2[2 * layer_idx + 1]  # weights at odd indices
            
            t1 = dict1[orig_weight_key]
            t2 = dict2[new_weight_key]
            
            # Handle last layer shape: original (512,2,2,10) vs new (10,2048)
            if layer_idx == 4:
                if t1.shape == torch.Size([512, 2, 2, 10]) and t2.shape == torch.Size([10, 2048]):
                    t1 = t1.permute(3, 0, 1, 2).reshape(10, -1)
            
            result = compare_tensors(t1, t2, f"{name_prefix}/weight_{layer_idx}", verbose=verbose)
            result['name'] = f"{name_prefix}/weight_{layer_idx}"
            results.append(result)
    else:
        # Standard comparison by index
        for i, (k1, k2) in enumerate(zip(keys1, keys2)):
            t1 = dict1[k1]
            t2 = dict2[k2]
            
            result = compare_tensors(t1, t2, f"{name_prefix}/{i}", verbose=verbose)
            result['name'] = f"{name_prefix}/{i}"
            result['key_orig'] = k1
            result['key_new'] = k2
            results.append(result)
    
    return results



def compare_energies(file1, file2, verbose=True):
    """Compare energy values."""
    if not os.path.exists(file1) or not os.path.exists(file2):
        if verbose:
            print(f"  [Energies] Missing files")
        return {'match': False, 'error': 'missing_file'}
    
    e1 = torch.load(file1, map_location='cpu', weights_only=True)
    e2 = torch.load(file2, map_location='cpu', weights_only=True)
    
    results = {}
    all_match = True
    
    for key in ['energy_free', 'energy_pos', 'energy_neg']:
        v1 = e1.get(key, float('nan'))
        v2 = e2.get(key, float('nan'))
        diff = abs(v1 - v2)
        match = diff < 1e-4
        all_match = all_match and match
        results[key] = {'orig': v1, 'new': v2, 'diff': diff, 'match': match}
        
        if verbose:
            status = "✓" if match else "✗"
            print(f"  [{key}] {status} Orig: {v1:.6f} | New: {v2:.6f} | Diff: {diff:.2e}")
    
    results['all_match'] = all_match
    return results


def find_batch_files(dump_dir):
    """Find all batch files in a dump directory."""
    pattern = os.path.join(dump_dir, 'batch_*_input.pt')
    files = glob.glob(pattern)
    batch_indices = []
    for f in files:
        # Extract batch index from filename
        basename = os.path.basename(f)
        idx = int(basename.split('_')[1])
        batch_indices.append(idx)
    return sorted(batch_indices)


def compare_batch(orig_dir, new_dir, batch_idx, verbose=True):
    """Compare all dumps for a single batch."""
    orig_prefix = os.path.join(orig_dir, f'batch_{batch_idx:05d}')
    new_prefix = os.path.join(new_dir, f'batch_{batch_idx:05d}')
    
    results = {
        'batch_idx': batch_idx,
        'comparisons': {}
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Batch {batch_idx}")
        print(f"{'='*60}")
    
    # # Compare inputs (should be identical)
    # if verbose:
    #     print("\n--- Inputs ---")
    # input_results = compare_dict_files(
    #     f'{orig_prefix}_input.pt', 
    #     f'{new_prefix}_input.pt',
    #     'input',
    #     verbose=verbose
    # )
    # results['comparisons']['input'] = input_results
    
    # Compare free states
    if verbose:
        print("\n--- Free States ---")
    free_results = compare_dict_files(
        f'{orig_prefix}_free_states.pt',
        f'{new_prefix}_free_states.pt', 
        'free_states',
        verbose=verbose
    )
    results['comparisons']['free_states'] = free_results
    
    # Compare positive nudged states
    if verbose:
        print("\n--- Positive Nudged States ---")
    pos_results = compare_dict_files(
        f'{orig_prefix}_pos_nudged_states.pt',
        f'{new_prefix}_pos_nudged_states.pt',
        'pos_nudged_states',
        verbose=verbose
    )
    results['comparisons']['pos_nudged_states'] = pos_results
    
    # Compare negative nudged states
    if verbose:
        print("\n--- Negative Nudged States ---")
    neg_results = compare_dict_files(
        f'{orig_prefix}_neg_nudged_states.pt',
        f'{new_prefix}_neg_nudged_states.pt',
        'neg_nudged_states',
        verbose=verbose
    )
    results['comparisons']['neg_nudged_states'] = neg_results
    
    # Compare energies
    if verbose:
        print("\n--- Energies ---")
    energy_results = compare_energies(
        f'{orig_prefix}_energies.pt',
        f'{new_prefix}_energies.pt',
        verbose=verbose
    )
    results['comparisons']['energies'] = energy_results
    
    # Compare gradients
    if verbose:
        print("\n--- Gradients ---")
    grad_results = compare_dict_files(
        f'{orig_prefix}_gradients.pt',
        f'{new_prefix}_gradients.pt',
        'gradients',
        verbose=verbose
    )
    results['comparisons']['gradients'] = grad_results
    
    # Compare weights before
    if verbose:
        print("\n--- Weights Before Update ---")
    weights_before_results = compare_dict_files(
        f'{orig_prefix}_weights_before.pt',
        f'{new_prefix}_weights_before.pt',
        'weights_before',
        verbose=verbose
    )
    results['comparisons']['weights_before'] = weights_before_results
    
    # Compare weights after
    if verbose:
        print("\n--- Weights After Update ---")
    weights_after_results = compare_dict_files(
        f'{orig_prefix}_weights_after.pt',
        f'{new_prefix}_weights_after.pt',
        'weights_after',
        verbose=verbose
    )
    results['comparisons']['weights_after'] = weights_after_results
    
    return results


def summarize_comparison(comparison_type, results_list, threshold=1e-4):
    """Summarize comparison results for a type across all batches."""
    max_diffs = []
    mismatches = []
    
    for i, results in enumerate(results_list):
        if isinstance(results, list):
            for r in results:
                if 'max_diff' in r:
                    max_diffs.append(r['max_diff'])
                    if r['max_diff'] > threshold:
                        mismatches.append((i, r.get('name', '?'), r['max_diff']))
        elif isinstance(results, dict):
            if 'max_diff' in results:
                max_diffs.append(results['max_diff'])
                if results['max_diff'] > threshold:
                    mismatches.append((i, comparison_type, results['max_diff']))
    
    if max_diffs:
        return {
            'max_max_diff': max(max_diffs),
            'mean_max_diff': np.mean(max_diffs),
            'num_mismatches': len(mismatches),
            'first_mismatch': mismatches[0] if mismatches else None
        }
    return None


def main():
    parser = argparse.ArgumentParser(description='Compare dumps from original and new EquiProp')
    parser.add_argument('--orig_dir', type=str, default='dumps/original', help='Original dumps directory')
    parser.add_argument('--new_dir', type=str, default='dumps/new', help='New dumps directory')
    parser.add_argument('--verbose', action='store_true', help='Print detailed comparisons')
    parser.add_argument('--threshold', type=float, default=1e-4, help='Threshold for considering a mismatch')
    parser.add_argument('--max_batches', type=int, default=None, help='Maximum batches to compare')
    args = parser.parse_args()
    
    print("="*70)
    print("EqProp Implementation Comparison")
    print("="*70)
    print(f"Original: {args.orig_dir}")
    print(f"New:      {args.new_dir}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Find batches to compare
    orig_batches = set(find_batch_files(args.orig_dir))
    new_batches = set(find_batch_files(args.new_dir))
    
    common_batches = sorted(orig_batches & new_batches)
    
    if args.max_batches:
        common_batches = common_batches[:args.max_batches]
    
    print(f"Found {len(orig_batches)} original batches, {len(new_batches)} new batches")
    print(f"Comparing {len(common_batches)} common batches")
    
    if not common_batches:
        print("\nNo common batches found! Make sure both scripts have been run.")
        return
    
    # Compare each batch
    all_results = []
    for batch_idx in common_batches:
        results = compare_batch(args.orig_dir, args.new_dir, batch_idx, verbose=args.verbose)
        all_results.append(results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    comparison_types = ['input', 'free_states', 'pos_nudged_states', 'neg_nudged_states', 
                        'gradients', 'weights_before', 'weights_after']
    
    first_divergence = None
    
    for comp_type in comparison_types:
        type_results = [r['comparisons'].get(comp_type, []) for r in all_results]
        summary = summarize_comparison(comp_type, type_results, args.threshold)
        
        if summary:
            status = "✓" if summary['num_mismatches'] == 0 else "✗"
            print(f"{status} {comp_type:20s} | Max Diff: {summary['max_max_diff']:.2e} | "
                  f"Mean Max Diff: {summary['mean_max_diff']:.2e} | Mismatches: {summary['num_mismatches']}")
            
            if summary['first_mismatch'] and first_divergence is None:
                first_divergence = (comp_type, summary['first_mismatch'])
    
    # Energy summary
    print("\n--- Energy Summary ---")
    for key in ['energy_free', 'energy_pos', 'energy_neg']:
        diffs = []
        for r in all_results:
            if 'energies' in r['comparisons'] and isinstance(r['comparisons']['energies'], dict):
                if key in r['comparisons']['energies']:
                    diffs.append(r['comparisons']['energies'][key]['diff'])
        
        if diffs:
            max_diff = max(diffs)
            mean_diff = np.mean(diffs)
            status = "✓" if max_diff < args.threshold else "✗"
            print(f"{status} {key:15s} | Max Diff: {max_diff:.2e} | Mean Diff: {mean_diff:.2e}")
    
    # First divergence
    if first_divergence:
        print(f"\n⚠️  First divergence found in '{first_divergence[0]}' at batch {first_divergence[1][0]}")
        print(f"   Component: {first_divergence[1][1]}, Max diff: {first_divergence[1][2]:.2e}")
    else:
        print("\n✓ All comparisons within threshold!")
    
    print()


if __name__ == '__main__':
    main()
