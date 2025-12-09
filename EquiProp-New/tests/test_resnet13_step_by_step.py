
import torch
import torch.nn as nn
import sys
import os
import numpy as np
import importlib

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(tests_dir)

from utils import sync_weights_resnet, compare_tensors, compare_gradients

# Path to Original EquiProp
root_dir = os.path.abspath(os.path.join(tests_dir, "../../../"))
equiprop_dir = os.path.join(root_dir, "Scaling-Equilibrium-Propagation-to-Deeper-Neural-Network-Architectures/EquiProp")

def load_new_resnet_model(device='cpu', activation=None):
    """Loads the new ResNet13_Interactions model."""

    import eqprop.interactions.core as new_core
    from eqprop.activation import relu6, hard_sigmoid
    ResNet13_Interactions = new_core.ResNet13_Interactions
    
    if activation == 'relu6':
        act_fn = relu6
    elif activation == 'hard-sigmoid':
        act_fn = hard_sigmoid
    else:
        act_fn = hard_sigmoid # Default
        
    model = ResNet13_Interactions(num_inputs=3, num_outputs=10, activation=act_fn)
    model.to(device)
    model.eval()
    
    return model


def load_original_resnet_full(device='cuda', batch_size=128):
    """
    Loads the original ResNet13 model and necessary components for step-by-step execution.
    """
    if equiprop_dir not in sys.path:
        sys.path.insert(0, equiprop_dir)
        
    try:
        import model.hopfield.network as orig_net
        import model.function.network as orig_func_net
        import model.hopfield.minimizer as orig_minimizer
        import training.sgd as orig_sgd
        import model.function.cost as orig_cost
        
        # Reload to ensure clean state
        importlib.reload(orig_net)
        importlib.reload(orig_func_net)
        importlib.reload(orig_minimizer)
        importlib.reload(orig_sgd)
        importlib.reload(orig_cost)
        
        ConvHopfieldResEnergy32 = orig_net.ConvHopfieldResEnergy32
        Network = orig_func_net.Network
        FixedPointMinimizer = orig_minimizer.FixedPointMinimizer
        AugmentedFunction = orig_sgd.AugmentedFunction
        CrossEntropy = orig_cost.CrossEntropy
        SquaredError = orig_cost.SquaredError
        EquilibriumProp = orig_sgd.EquilibriumProp
        
        # Params
        weight_gains = [
            0.6, 0.6, 0.7,   # block-1
            0.6, 0.7, 0.6,   # block-2
            0.6, 0.7, 0.6,   # block-3
            0.6, 0.7, 0.6,   # block-4
            0.8              # dense head
        ]
        activation = 'relu6'
        
        # Create Model (Energy Function)
        energy_fn = ConvHopfieldResEnergy32(3, 10, weight_gains=weight_gains, activation=activation)
        energy_fn.set_device(device)
        
        # Create Network Wrapper
        network = Network(energy_fn)
        
        # Cost Function
        output_layer = energy_fn.layers()[-1]
        # cost_fn = CrossEntropy(output_layer)
        cost_fn = SquaredError(output_layer)
        
        # Augmented Function (Energy + Cost)
        augmented_fn = AugmentedFunction(energy_fn, cost_fn)
        
        # Minimizers
        free_layers = network.free_layers()
        
        # Inference Minimizer (Hopfield Energy only)
        minimizer_inference = FixedPointMinimizer(energy_fn, free_layers)
        minimizer_inference.mode = 'synchronous' # Use synchronous for deterministic step-by-step comparison
        
        # Training Minimizer (Augmented Energy)
        minimizer_training = FixedPointMinimizer(augmented_fn, free_layers)
        minimizer_training.mode = 'synchronous'
        
        # Estimator (for gradients)
        params = energy_fn.params()
        layers = energy_fn.layers()
        estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, minimizer_training)
        estimator.variant = 'centered'
        
        return {
            'model': energy_fn,
            'network': network,
            'minimizer_inference': minimizer_inference,
            'minimizer_training': minimizer_training,
            'augmented_fn': augmented_fn,
            'cost_fn': cost_fn,
            'estimator': estimator,
            'modules': {
                'orig_net': orig_net,
                'orig_sgd': orig_sgd
            }
        }
        
    except ImportError as e:
        print(f"Error importing original code: {e}")
        raise e

def test_step_by_step():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    # Deterministic settings
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # Disable TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # 1. Load Models
    print("Loading Original Model...")
    orig_bundle = load_original_resnet_full(device)
    orig_model = orig_bundle['model']
    orig_network = orig_bundle['network']
    minimizer_infer = orig_bundle['minimizer_inference']
    minimizer_train = orig_bundle['minimizer_training']
    augmented_fn = orig_bundle['augmented_fn']
    cost_fn = orig_bundle['cost_fn']
    
            
    print("Loading New Model...")
    new_model = load_new_resnet_model(device=device, activation='relu6')
    new_model.cost_type = 'MSE'
    
    # 2. Sync Weights
    print("Syncing Weights...")
    sync_weights_resnet(orig_model, new_model)
    
    # 3. Setup Inputs
    B = 10 # Batch size
    x = torch.randn(B, 3, 32, 32, device=device)
    y = torch.randint(0, 10, (B,), device=device)
    
    # Set inputs
    orig_network.set_input(x)
    cost_fn.set_target(y) # For cost calculation
    
    # 4. Initialize States
    print("Initializing States...")
    for layer in orig_model.layers():
        if layer != orig_model.layers()[0]: # Skip input layer
            layer.init_state(B, device)
            
    # New: create_states returns list of zeros
    new_states = new_model.create_states(B, device)
    
    orig_layers = orig_model.layers()
    # Map: orig_layers[i+1] <-> new_states[i]
    
    # 5. Free Phase Loop (120 steps)
    print("\n--- Starting Free Phase (120 steps) ---")
    n_free = 120
    beta = 0.0
    
    for step in range(n_free):
        # Original Step
        minimizer_infer.step(minimizer_infer._updaters)
        
        # New Step
        new_states = new_model.minimize_step(x, new_states, beta=0.0, target=y) 
        
        # Compare
        print(f"Step {step+1}:")
        all_match = True
        for i in range(len(new_states)):
            orig_s = orig_layers[i+1].state
            new_s = new_states[i]
            
            if orig_s.shape != new_s.shape:
                 if orig_s.numel() == new_s.numel():
                     orig_s = orig_s.view_as(new_s)
            
            compare_tensors(orig_s, new_s, f"Free State {i+1}")
            # breakpoint()
            
    print("Free Phase Complete.")

    orig_free_states = [l.state.detach().clone() for l in orig_model.layers()]
    new_free_states = [s.detach().clone() for s in new_states]
    
    print("\n--- Starting Nudged Phase (50 steps) ---")
    n_nudge = 50
    beta = 0.1 # Nudging factor
    
    augmented_fn.nudging = beta
    
    for step in range(n_nudge):
        print(f"Step {step+1}")
        minimizer_train.step(minimizer_train._updaters)
        
        # New Step
        new_states = new_model.minimize_step(x, new_states, beta=beta, target=y)

        for i in range(len(new_states)):
            orig_s = orig_layers[i+1].state
            new_s = new_states[i]
            
            if orig_s.shape != new_s.shape:
                 if orig_s.numel() == new_s.numel():
                     orig_s = orig_s.view_as(new_s)
            
            compare_tensors(orig_s, new_s, f"Nudged State {i+1}")
            # breakpoint()
        
    new_pos_nudged_states = [s.detach().clone() for s in new_states]

    print("\n--- Starting Negative Nudged Phase (50 steps) ---")
    beta_neg = -0.1

    for layer, state in zip(orig_model.layers()[1:], orig_free_states[1:]):
        layer.state = state.detach().clone()  
    new_states = [s.clone() for s in new_free_states]
    
    augmented_fn.nudging = beta_neg
    
    for step in range(n_nudge):
        print(f"Step {step+1}")
        minimizer_train.step(minimizer_train._updaters)
        new_states = new_model.minimize_step(x, new_states, beta=beta_neg, target=y)
        
        for i in range(len(new_states)):
            orig_s = orig_layers[i+1].state
            new_s = new_states[i]
            
            if orig_s.shape != new_s.shape:
                 if orig_s.numel() == new_s.numel():
                     orig_s = orig_s.view_as(new_s)
            
            compare_tensors(orig_s, new_s, f"Negative Nudged State {i+1}")

    new_neg_nudged_states = [s.detach().clone() for s in new_states]
            
    print("\n--- Checking Gradients ---")
    
    for layer, state in zip(orig_model.layers()[1:], orig_free_states[1:]):
        layer.state = state.detach().clone()
    
    orig_bundle['estimator'].variant = 'centered'
    orig_bundle['estimator'].nudging = beta

    print("Computing Original Gradients (Centered)...")
    orig_bundle['estimator']._energy_minimizer.num_iterations = 50
    orig_grads = orig_bundle['estimator'].compute_gradient()

    
    
    new_energy_pos = new_model.energy(x, new_pos_nudged_states).mean()
    new_energy_neg = new_model.energy(x, new_neg_nudged_states).mean()

    grads_pos = torch.autograd.grad(new_energy_pos, new_model.parameters())
    grads_neg = torch.autograd.grad(new_energy_neg, new_model.parameters())
    
    # Combine: (Grad_Pos - Grad_Neg) / (2*beta)
    new_grads = [(gp - gn) / (2*beta) for gp, gn in zip(grads_pos, grads_neg)]

    # Apply Gradient Clipping to Match Original Model
    grad_clip = 1.0
    total_norm = torch.norm(torch.stack([g.norm() for g in new_grads]))
    if total_norm > grad_clip:
        scale = grad_clip / (total_norm + 1e-6)
        new_grads = [g * scale for g in new_grads]
        print(f"Applied Gradient Clipping to New Gradients (Scale: {scale:.4f})")
    

    # orig_grads order: [Bias1, ..., Bias9, Weight1, ..., Weight13] (assuming params order)
    
    # Map
    # Biases: 0-8
    # Weights: 9-21
    
    new_grads_map = {}
    grad_iter = iter(new_grads)
    
    # Define Maps
    bias_positions = [0, 1, 3, 5, 6, 8, 9, 11, 12]
    weight_positions = [i for i in range(13)]

    orig_grads_map = {}
    for i, orig_grad in enumerate(orig_grads):
        if i<9:
            # 0, 1, 3, 4, 6, 7, 9, 10, 12
            orig_grads_map[bias_positions[i]] = {}
            if bias_positions[i] == 12:
                orig_grads_map[bias_positions[i]]['bias'] = orig_grads[i]
            else:
                orig_grads_map[bias_positions[i]]['bias'] = orig_grads[i].unsqueeze(-1).unsqueeze(-1)
        else:
            if i-9 not in bias_positions:
                orig_grads_map[i-9] = {}
            orig_grads_map[i-9]['weight'] = orig_grads[i]
    
    # Fix Weight 12 (Linear)
    # Original: (512, 2, 2, 10) -> (C, H, W, Out)
    # New: (10, 2048) -> (Out, C*H*W)
    # We need to permute Original to (Out, C, H, W) and then flatten to (Out, C*H*W)
    # orig_grads_map[12]['weight'] is (512, 2, 2, 10)
    orig_grads_map[12]['weight'] = orig_grads_map[12]['weight'].permute(3, 0, 1, 2).reshape(10, -1)
    
    for i, interaction in enumerate(new_model.interactions):
        new_grads_map[i] = {}

        if hasattr(interaction, 'bias') and isinstance(interaction.bias, torch.nn.Parameter):
            new_grads_map[i]['bias'] = next(grad_iter)
            
        if hasattr(interaction, 'conv'):
            new_grads_map[i]['weight'] = next(grad_iter)
        elif hasattr(interaction, 'linear'):
            new_grads_map[i]['weight'] = next(grad_iter)
            
    print("\n--- Verifying Gradient Norms ---")
    orig_norm = torch.norm(torch.stack([g.norm() for g in orig_grads]))
    new_norm = torch.norm(torch.stack([g.norm() for g in new_grads]))
    
    print(f"Original Gradient Total Norm: {orig_norm:.4f}")
    print(f"New Gradient Total Norm: {new_norm:.4f}")
    
    # Compare Gradients with better metrics
    print("\n" + "="*60)
    print("Comparing Gradients (with Cosine Similarity & Normalized Error):")
    print("="*60)
    
    all_results = []
    for i in sorted(orig_grads_map.keys()):
        if 'bias' in orig_grads_map[i]:
            result = compare_gradients(orig_grads_map[i]['bias'], new_grads_map[i]['bias'], f"Bias {i}")
            all_results.append((f"Bias {i}", result))
        result = compare_gradients(orig_grads_map[i]['weight'], new_grads_map[i]['weight'], f"Weight {i}")
        all_results.append((f"Weight {i}", result))
    
    # Summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    matches = sum(1 for _, r in all_results if r.get('match', False))
    total = len(all_results)
    print(f"Matched: {matches}/{total}")
    
    # Average cosine similarity
    cosines = [r['cosine_similarity'] for _, r in all_results if 'cosine_similarity' in r]
    if cosines:
        print(f"Mean Cosine Similarity: {sum(cosines)/len(cosines):.6f}")
        print(f"Min Cosine Similarity:  {min(cosines):.6f}")

    breakpoint()

if __name__ == "__main__":
    test_step_by_step()
