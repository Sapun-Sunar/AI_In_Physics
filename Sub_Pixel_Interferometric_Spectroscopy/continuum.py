import numpy as np
import matplotlib.pyplot as plt

def generate_lab_continuum(rows=250, cols=400):
    """
    Phase 1 & 2: Spawns the custom 250x400 landscape matrix and fills it with 
    the true-to-life plasma continuum envelope.
    """
    center_x = np.round(np.random.uniform(180,220),8)
    center_y = np.round(np.random.uniform(80,120),8)
    # --- 1. THE PITCH-DARK VOID ---
    y_indices, x_indices = np.indices((rows, cols))
    
    # --- 2. THE RADIAL TOPOGRAPHY ---
    radii_squared = (x_indices - center_x)**2 + (y_indices - center_y)**2
    
    # --- 3. THE THERMAL ENVELOPE (2D Gaussian) ---
    plasma_spread = 15000#np.round(np.random.uniform(10000,20000),8)
    base_intensity = 450 * np.exp(-radii_squared / plasma_spread) #np.round(np.random.uniform(400,550),8)
    
    # --- 4. THE ASYMMETRICAL SLANT ---
    
    slant_gradient = 1# 1.0 - (x_indices * np.round(np.random.uniform(0.0005,0.0001),8)) - (y_indices * np.round(np.random.uniform(0.0008,0.001),8))
    plasma_continuum = base_intensity * slant_gradient
    # Clip to absolute physical limits
    plasma_continuum = np.clip(plasma_continuum, 0.0, None)
    
    return plasma_continuum, center_x, center_y

if __name__ == "__main__":
    print("Spawning the 250x400 physical universe...")
    plasma_field,_,_ = generate_lab_continuum()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(plasma_field, cmap='inferno', origin='lower')
    plt.title("Phase 1 & 2: Synthetic Plasma Continuum\n(250 Rows x 400 Columns)")
    plt.colorbar(label="Photon Count")
    plt.xlabel("Columns (X-Axis)")
    plt.ylabel("Rows (Y-Axis)")
    plt.tight_layout()
    plt.show()