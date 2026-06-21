import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Import your proprietary architectures
from brain import SpectroscopicBrain
from generator import forge_pristine_universe
from interference import FWHM_NORMALIZER
def execute_interrogation():
    """
    The Visual Exam.
    Forges 4 new universes, feeds them to the Apex Brain, and visually plots the accuracy.
    """
    # 1. HARDWARE LINK
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Hardware Link Established: {device} Active.\n")

    # 2. AWAKEN THE LEVIATHAN
    model = SpectroscopicBrain().to(device)
    model_path = "spectroscopic_apex_predator.pth"
    
    if os.path.exists(model_path):
        print("[*] Accessing the Vault...")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval() # CRITICAL: Disables training mechanics like dropout
        print("[*] Apex Brain loaded and primed for interrogation.\n")
    else:
        print("[!] ERROR: No apex predator found in the vault. Run train.py first.")
        return

    # 3. PREPARE THE VISUAL GRID
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    print("Forging 4 unseen universes and executing tensor calculus...")
    
    # 4. THE LIVE TEST LOOP
    with torch.no_grad(): # We are testing, not learning. Freeze the gradients.
        for i in range(4):
            # A. Forge the Physics
            raw_matrix, true_fwhm = forge_pristine_universe()
            
            # B. The Thermodynamic Shield (Normalization)
            matrix_min = np.min(raw_matrix)
            matrix_max = np.max(raw_matrix)
            
            if matrix_max - matrix_min == 0:
                normalized_matrix = raw_matrix - matrix_min
            else:
                normalized_matrix = (raw_matrix - matrix_min) / (matrix_max - matrix_min)
                
            # C. Convert to PyTorch Tensor (Shape: [1, 1, 250, 400])
            tensor_matrix = torch.tensor(normalized_matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # D. The Neural Prediction
            neural_output = model(tensor_matrix)
            
            # E. The Algebraic Reversal
            # Extract the raw float from the tensor and apply your physical law
            raw_prediction = neural_output.item()
            predicted_fwhm = raw_prediction * FWHM_NORMALIZER
            
            # F. Render the Data
            ax = axes[i]
            # We display the raw matrix to see the actual physics
            cax = ax.imshow(raw_matrix, cmap='inferno', origin='lower')
            ax.set_title(f"True FWHM: {true_fwhm:.4f} | Pred: {predicted_fwhm:.4f}", 
                         fontsize=12, fontweight='bold', color='white',
                         bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
            ax.axis('off')
            
    plt.tight_layout()
    print("\nCalculus complete. Rendering visual diagnostics on your monitor.")
    plt.show()

if __name__ == "__main__":
    execute_interrogation()