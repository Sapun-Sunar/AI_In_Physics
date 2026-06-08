import torch
import numpy as np
import matplotlib.pyplot as plt
from brain import Spectroscopic_Brain

def photograph_ai_thoughts(matrix_path, model_path):
    print("Initiating Saliency Brain-Scan...")

    # Extracting the absolute true answer from the filename string.
    parts = matrix_path.replace(".npy", "").split("_")
    true_fwhm = float(parts[-1])


    # 1. Hardware Targeting.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA Cuda.")
    else:
        device = torch.device("cpu")


    # 2. Awaken the Brain.
    brain = Spectroscopic_Brain().to(device)
    brain.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
    brain.eval() # Lock the weights


    # 3. Ingest the data.
    raw_matrix = np.load(matrix_path)
    # CRITICAL: We tell PyTorch to track the calculus of the image itself
    tensor_image = torch.tensor(raw_matrix, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    tensor_image.requires_grad_() 


    # 4. The Forward Execution.
    prediction = brain(tensor_image)
    

    # 5. The Backward Autopsy (Tracing the thoughts)
    # We violently pull the derivative of the prediction back to the pixels
    prediction.backward()
    

    # Extract the absolute value of the gradients to create the Heatmap
    saliency_map = tensor_image.grad.data.abs().squeeze().cpu().numpy()
    

    # Clean the map: normalize it so the brightest thought is 1.0
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())



    # Visualization.
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    
    # Left Side: The raw, data / matrix used.
    axes[0].imshow(raw_matrix, cmap='magma')
    axes[0].set_title(f'Raw Sensor Data (True FWHM: {true_fwhm:.4f})', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Right Side: The AI's Visual Cortex
    # We overlay the glowing red thoughts on top of the dark image
    axes[1].imshow(raw_matrix, cmap='gray')
    im = axes[1].imshow(saliency_map, cmap='inferno', alpha=0.6) # Alpha makes it a glowing overlay
    axes[1].set_title(f'AI Saliency Map (Predicted FWHM: {prediction.item():.4f})', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('Visual_Brain_Scan.png')
    print("SUCCESS: The thoughts of the machine have been photographed.")
    plt.show()



if __name__ == "__main__":

    matrix_path = "Your_file_Location / the one created from data_creation.py"
    model_path = "Your_model_path / simply writing the model name which is in this directory would work."
    photograph_ai_thoughts(matrix_path, model_path)