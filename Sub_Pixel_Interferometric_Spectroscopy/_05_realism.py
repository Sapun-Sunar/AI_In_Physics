import numpy as np
import matplotlib.pyplot as plt

def apply_optical_aberrations(clean_matrix, center_x=200, center_y=125, safe_radius=130.0):
    """
    Phase 3.5: Injects physical lens defects, stray light flares, and sensor obstructions.
    The 'safe_radius' acts as a mathematical shield, protecting the inner core.
    """
    rows, cols = clean_matrix.shape
    y_indices, x_indices = np.indices((rows, cols))
    radii = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
    
    # --- 1. THE VULNERABILITY MASK (The Aegis Shield) ---
    # 0.0 means totally protected (inner rings). 1.0 means totally vulnerable (outer rings).
    # The / 50.0 creates a smooth, natural fade rather than a harsh cut-off line.
    vulnerability_mask = np.clip((radii - safe_radius) / 50.0, 0.0, 1.0)
    
    # --- 2. LENS MISALIGNMENT (Stray Light Flare) ---
    # A massive burst of stray photons blooming in the top right corner
    flare_x, flare_y = 350, 200 
    flare_radius_sq = (x_indices - flare_x)**2 + (y_indices - flare_y)**2
    flare_intensity = 350.0 * np.exp(-flare_radius_sq / 15000.0) 
    
    # The flare only appears where the shield is weak
    stray_light = flare_intensity * vulnerability_mask
    
    # --- 3. SENSOR OBSTRUCTIONS (The Smeared Black Parts) ---
    # We drop 4 to 8 random dark smears across the matrix
    obstruction_map = np.ones((rows, cols))
    num_smears = np.random.randint(4, 9)
    
    for _ in range(num_smears):
        smear_x = np.random.randint(0, cols)
        smear_y = np.random.randint(0, rows)
        smear_width = np.random.uniform(1000, 4000) # How fat the smear is
        depth = np.random.uniform(0.3, 0.8)         # Eats 30% to 80% of light
        
        smear_dist = (x_indices - smear_x)**2 + (y_indices - smear_y)**2
        dip = 1.0 - (depth * np.exp(-smear_dist / smear_width))
        obstruction_map *= dip
        
    # We blend the smears using the vulnerability mask. 
    # If vulnerability is 0, the transmission stays at 1.0 (perfect glass).
    final_transmission = 1.0 - (vulnerability_mask * (1.0 - obstruction_map))
    
    # --- SYNTHESIS ---
    # Add the stray flare, then choke the light with the obstructions
    aberrated_matrix = (clean_matrix + stray_light) * final_transmission
    
    return aberrated_matrix

if __name__ == "__main__":
    # A quick visual test of the aberration engine
    print("Testing Optical Aberrations...")
    test_grid = np.ones((250, 400)) * 700.0 # Flat bright background
    ruined_grid = apply_optical_aberrations(test_grid)
    
    plt.figure(figsize=(8, 5))
    plt.imshow(ruined_grid, cmap='plasma', vmin=400, vmax=900)
    plt.title("Isolated Output of realism.py\nNotice the protected center core and ruined edges.")
    plt.colorbar()
    plt.show()