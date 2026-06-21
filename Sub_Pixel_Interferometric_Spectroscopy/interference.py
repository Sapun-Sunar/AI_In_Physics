import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
# Import the light source from your first module
from continuum import generate_lab_continuum

# [!] Universal Constant for Sigmoid Sweet Spot Normalization
FWHM_NORMALIZER = 10.0

def apply_interference(continuum_matrix, center_x, center_y):
    """
    Phase 3: Generates the Airy transmission mask and modulates the continuum.
    """
    rows, cols = continuum_matrix.shape
    y_indices, x_indices = np.indices((rows, cols))
    
    # 1. Radial Geometry
    radii_squared = (x_indices - center_x)**2 + (y_indices - center_y)**2
    radii = np.sqrt(radii_squared)
    
    # 2. The Airy Transmission Mask (Values from 0.0 to 1.0)
    # Finesse controls the sharpness of the bright fringes
    finesse = np.round(np.random.uniform(2,5),8) #8
    # Fringe spacing controls how fast the rings compress outward
    fringe_spacing = 0.0007#np.round(np.random.uniform(0.0005,0.0009),4)
    
    phase = fringe_spacing * radii_squared
    
    # Transmission = 1 / (1 + F * sin^2(phase/2))
    transmission_mask = 1.0 / (1.0 + finesse * np.sin(phase)**2)
    transmission_mask = gaussian_filter(transmission_mask, sigma=np.round(np.random.uniform(3,4),8))#7 
    
    # [!] The True Physical Scale: Absorption dips are 3.0 to 7.0 pixels wide
    true_fwhm = np.round(np.random.uniform(3.0, 7.0), 8)
    base_gamma = true_fwhm / 2.0
    
    # Carve the Lorentzian mask
    total_lorentzian_dip = np.zeros_like(radii)
    base_absorption_depth = 0.6#np.round(np.random.uniform(0.3,0.6),4)
    
    # --- THE ADAPTABLE LOGIC ---
    # Find the absolute furthest corner of the sensor matrix
    max_radius_squared = np.max(radii_squared)
    
    # Calculate exactly how many rings fit inside this universe
    max_m = int(np.floor((max_radius_squared * fringe_spacing) / np.pi))
    
    # Fallback: Ensure at least the first ring is targeted if heavily zoomed in
    max_m = max(1, max_m) 
    
    # [!] RESTORED m=0: Carve the missing shadow directly over the central plasma bullseye
    for m in range(0, max_m + 1):
        # If m=0, the target radius is exactly 0.0 (dead center)
        target_r = np.sqrt((m * np.pi) / fringe_spacing) if m > 0 else 0.0
        
        # Decay depth: 100%, 75%, 56%, etc., as they expand outward
        current_depth = base_absorption_depth * (0.75 ** m)
        current_gamma = base_gamma / np.sqrt(m + 1)
        
        # Carve the shadow
        dip = (current_depth * current_gamma**2) / ((radii - target_r)**2 + current_gamma**2)
        total_lorentzian_dip += dip
        
    # Ensure physical limits are respected
    total_lorentzian_dip = np.clip(total_lorentzian_dip, 0.0, 1.0)
    absorption_mask = 1.0 - total_lorentzian_dip
    
    # 4. Optical Modulation
    # Plasma * Fabry-Perot Rings * Cold Gas Absorption
    interference_pattern = continuum_matrix * transmission_mask * absorption_mask
    
    
    # 4. The Sensor Dark Current (The Missing Physics)
    # We add the ~450 baseline voltage that the camera naturally produces
    dark_current_offset = 450#np.round(np.random.uniform(380,450),4)
    final_sensor_readout = interference_pattern + dark_current_offset
    
    return final_sensor_readout, true_fwhm

if __name__ == "__main__":
    print("Linking modules... Modulating continuum with Fabry-Pérot mechanics.")
    
    # Pull the light source
    #plasma_light,center_x, center_y = generate_lab_continuum()
    
    # Pass it through the interferometer
    ring_universe, true_fwhm = apply_interference(*generate_lab_continuum())
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(8, 6))
    plt.imshow(ring_universe, cmap='inferno', origin='lower', vmin=450, vmax=950)
    plt.title(f"The true fwhm is: {true_fwhm}")
    plt.colorbar(label="Sensor Intensity (Includes Dark Current)")
    plt.xlabel("Columns (X-Axis)")
    plt.ylabel("Rows (Y-Axis)")
    plt.tight_layout()
    plt.show()