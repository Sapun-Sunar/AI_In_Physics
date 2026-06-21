# Import the foundational physics modules
from continuum import generate_lab_continuum
from interference import apply_interference
from noise import apply_sensor_noise
import matplotlib.pyplot as plt


def forge_pristine_universe():
    """
    The Live Feeder Machine.
    Forges a universe strictly bound by quantum optics and baseline Poisson noise.
    Returns: A 2D numpy array (250x400) and its corresponding FWHM scalar.
    """
    # 1. Spawn the continuum plasma
    plasma_light,center_x,center_y = generate_lab_continuum()
    
    # 2. Project the wave mechanics and carve the absorption shadow
    clean_optics, true_fwhm = apply_interference(plasma_light,center_x,center_y)
    
    # 3. Inject Microscopic Sensor Noise directly onto the clean optics
    noisy_matrix= apply_sensor_noise(clean_optics)
    
    return noisy_matrix, true_fwhm

if __name__ == "__main__":
    # A quick diagnostic ping to ensure the Feeder is operational
    matrix, fwhm = forge_pristine_universe()
    plt.figure(figsize=(10,8))
    plt.imshow(matrix,cmap="inferno", origin="lower")
    plt.colorbar()
    plt.title("Synthetic Fabry-Perot Pattern.")
    plt.tight_layout()
    plt.show()