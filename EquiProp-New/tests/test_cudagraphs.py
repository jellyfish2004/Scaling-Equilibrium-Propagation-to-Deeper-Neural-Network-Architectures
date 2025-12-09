import torch
import sys
import os
import copy
from utils import compare_tensors

# Add path to EquiProp-New
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

from eqprop.interactions.core import ConvHopfieldEnergy32_Interactions

def test_cudagraphs():
    if not torch.cuda.is_available():
        print("Skipping cudagraphs test (CUDA not available)")
        return

    device = 'cuda'
    print(f"Running on {device}")
    
    # Enable deterministic mode
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Initialize model
    model = ConvHopfieldEnergy32_Interactions()
    model.to(device)
    model.eval()
    
    # Create a copy for cudagraphs to ensure no side effects
    model_cg = copy.deepcopy(model)
    model_cg.to(device)
    model_cg.eval()
    
    print("Compiling model with cudagraphs...")
    model_cg.minimize_step = torch.compile(model_cg.minimize_step, options={
        "epilogue_fusion": True,
        "max_autotune": True,
        "triton.cudagraphs": True,
    })
    
    # Inputs
    B = 4
    x = torch.randn(B, 3, 32, 32, device=device)
    
    # Initial states (same for both)
    states = model.create_states(B, device)
    states_cg = [s.clone() for s in states]
    
    # Run Baseline (No Compile)
    print("Running Baseline...")
    n_steps = 5
    beta = 0.0
    
    for _ in range(n_steps):
        states = model.minimize_step(x, states, beta=beta)
        
    # Run Cudagraphs
    print("Running Cudagraphs...")
    for _ in range(3):
        model_cg.minimize_step(x, states_cg, beta=beta)
        
    # Actual run
    for _ in range(n_steps):        
        new_states = model_cg.minimize_step(x, states_cg, beta=beta)
        states_cg = [s.clone() for s in new_states]
        
    # Compare Final States
    print("Comparing states...")
    all_match = True
    
    for i, (s, s_cg) in enumerate(zip(states, states_cg)):
        is_final = (i == len(states) - 1)
        
        if not compare_tensors(s, s_cg, f"State {i+1}"):
            if is_final:
                all_match = False
            
    if all_match:
        print("SUCCESS: Cudagraphs output matches baseline (within tolerance).")
    else:
        print("FAILURE: Cudagraphs output diverges significantly.")


if __name__ == "__main__":
    test_cudagraphs()
