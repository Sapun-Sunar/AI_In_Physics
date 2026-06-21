import numpy as np
import matplotlib.pyplot as plt

def apply_sensor_noise(clean_matrix):
    """
    Ingests the perfect optical matrix and applies real-world sensor degradation.
    """
    # 1. QUANTUM SHOT NOISE (Poisson)
    # Ensure no negative values before applying Poisson distribution
    safe_matrix = np.clip(clean_matrix, 0.0, None)
    
    # Generate the shot noise and lock it to float32 for Apple Silicon efficiency
    noisy_matrix = np.random.poisson(safe_matrix).astype(np.float64)
    
    # 2. THERMAL READ NOISE (Gaussian)
    # Simulates the baseline heat of the physical camera sensor
    thermal_noise = np.random.normal(loc=0.0, scale=np.round(np.random.uniform(2,3),8), size=noisy_matrix.shape).astype(np.float64) #4
    
    # Combine the physics
    final_degraded_matrix = noisy_matrix + thermal_noise
    
    return final_degraded_matrix

if __name__ == "__main__":
    print("Testing the noise quarantine...")
    # Generate a dummy smooth gradient to test the module
    test_gradient = np.linspace(500, 800, 250*400).reshape(250, 400)
    pixelated_gradient = apply_sensor_noise(test_gradient)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.imshow(test_gradient, cmap='inferno', vmin=400, vmax=900)
    ax1.set_title("Input: Pure Math")
    ax2.imshow(pixelated_gradient, cmap='inferno', vmin=400, vmax=900)
    ax2.set_title("Output: Pixelated Lab Reality")
    plt.show()