import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt


class AbsoluteVoidForge(Dataset):

    def __init__(self, num_samples=1000, matrix_size=300):
        """
        The Pure Optical Sandbox Matrix.
        Relies strictly on natural inverse-square physics for light decay.
        """
        self.num_samples = num_samples
        self.matrix_size = matrix_size
        
        # The physical width of the dark trench (our target for the AI)
        self.fwhm_range = (1, 2)


    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        true_fwhm = np.random.uniform(*self.fwhm_range)

        # 1. Literal Pixel Canvas & Optical Axis
        x = np.arange(0, self.matrix_size)
        y = np.arange(0, self.matrix_size)
        X, Y = np.meshgrid(x, y)

        center_x = self.matrix_size / 2.0 + np.random.uniform(-2, 2)
        center_y = self.matrix_size / 2.0 + np.random.uniform(-2, 2)
        R_base = np.sqrt((X - center_x)**2 + (Y - center_y)**2)




        # 2. THE CONTROL DESK

        # A. CONTINUUM GLOW
        glow_peak_intensity = np.round(np.random.uniform(0.2,0.5),1)
        glow_macroscopic_spread = np.random.randint(45,75)  # How wide the smooth nebula is
        glow_microscopic_scatter = np.random.randint(10,15)  # How jagged the edges of the nebula are
        
        barrier_start_radius = np.random.randint(80,100)  # Where the first partial barrier begins
        barrier_step_width = np.random.randint(10,20)    # Distance between each subsequent barrier ("each turn")
        transmission_rate = np.round(np.random.uniform(0.4,0.7),1)   # 80% passes through, 20% is destroyed per turn
        R_continuum = np.abs(R_base + np.random.normal(0, glow_microscopic_scatter, R_base.shape))


        # B. BRIGHT RING
        bright_ring_radius = np.random.randint(45,70)
        bright_ring_thickness = np.random.randint(5,30)    # The width of the bright light
        bright_scatter = np.random.uniform(1,3)          # Jaggedness of the bright ring
        R_bright = np.abs(R_base + np.random.normal(0, bright_scatter, R_base.shape))


        # C. DARK RING (THE VOID) 
        dark_ring_radius = bright_ring_radius + np.random.uniform(-1,1)   # How wide the pitch-black trench is
        dark_scatter = np.round(np.random.uniform(0.5,0.8),1)      # Jaggedness of the trench walls
        R_dark = np.abs(R_base + np.random.normal(0, dark_scatter, R_base.shape))
        # ---------------------------------------------------------



        # 3. THE EXECUTION 
        # A. Continuum Math (Base Inverse-Square)
        continuum_layer = glow_peak_intensity / (1.0 + (R_continuum / glow_macroscopic_spread)**2)

        # Apply the Discrete Partial Barriers
        # We calculate 'n': the number of barriers the light has physically crossed.
        # If R is less than the start radius, it has crossed 0 barriers.
        barriers_crossed = np.where(
            R_continuum > barrier_start_radius,
            np.floor((R_continuum - barrier_start_radius) / barrier_step_width) + 1,
            0
        )

        # Calculate the toll: (0.80) to the power of (barriers_crossed)
        # e.g., 0 crossed = 100% survives. 1 crossed = 80%. 2 crossed = 64%. 3 crossed = 51.2%.
        partial_transmission_mask = transmission_rate ** barriers_crossed
        
        # Execute the discrete step-down
        continuum_layer *= partial_transmission_mask



        # B. Bright Ring Math
        bright_layer = 0.8 * np.exp(-((R_bright - bright_ring_radius)**2) / (bright_ring_thickness**2))

        # C. Total Light
        total_light = continuum_layer + bright_layer

        # D. The Void Mask
        gaussian_width = true_fwhm / (2.0 * np.sqrt(np.log(2)))
        
        # We now use the converted gaussian_width to carve the exact mathematical void
        dark_mask = 1.0 - np.exp(-((R_dark - dark_ring_radius)**2) / (gaussian_width**2))





        # 4. SENSOR DAMAGE SIMULATOR (50/50) 
   
        damage_mask = np.ones_like(R_base)
        
        # 50% chance to generate a perfect ring, 50% chance to break it
        if np.random.rand() > 0.5:
            theta = np.arctan2(Y - center_y, X - center_x)
            start_angle = np.random.uniform(-np.pi, np.pi)
            cut_width = np.random.uniform(np.pi/4, 2.8) # Width of the occlusion
            end_angle = start_angle + cut_width
            
            # The calculus required to handle the 360-degree wrap-around
            if end_angle > np.pi: 
                damage_mask[(theta > start_angle) | (theta < end_angle - 2*np.pi)] = 0.0
            else:
                damage_mask[(theta > start_angle) & (theta < end_angle)] = 0.0

        # Apply the destruction (or survival) to the light matrix
        total_light *= damage_mask
        # ---------------------------------------------------------



        # 5. Final Cut
        matrix = total_light * dark_mask



        # 6. Final Clipping and Tensor Formatting
        final_matrix = np.clip(matrix, 0.0, 1.0).astype(np.float32)
        matrix_tensor = torch.tensor(final_matrix).unsqueeze(0)
        matrix_tensor = matrix_tensor[:, 30:275, 30:275]
        label_tensor = torch.tensor([true_fwhm], dtype=torch.float32)

        return matrix_tensor, label_tensor



# Saving some matrixes.
if __name__ == "__main__":
    print("Igniting the Physical Data Extractor...")
    
    # Define how many matrices you want.
    num_test_samples = 20
    dataset = AbsoluteVoidForge(num_samples=num_test_samples, matrix_size=300)
    
    # 1. Create a secure vault for the telemetry
    output_dir = "fabry_perot_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Extract and lock the pure mathematics
    for i in range(num_test_samples):
        # Pull the tensor and the answer from the Forge
        img_tensor, label_tensor = dataset[i]
        
        # Strip away the PyTorch formatting to get the raw NumPy matrix
        raw_matrix = img_tensor.squeeze().numpy()
        
        # Extract the absolute true answer
        true_fwhm = label_tensor[0].item()
        
        # 3. Save the matrix with the answer permanently stamped into the filename
        # Example filename: sample_003_fwhm_1.4523.npy
        file_path = os.path.join(output_dir, f"sample_{i:03d}_fwhm_{true_fwhm:.4f}.npy")
        np.save(file_path, raw_matrix)

        
    print(f"SUCCESS: {num_test_samples} pure matrices locked into the '{output_dir}' directory.") 