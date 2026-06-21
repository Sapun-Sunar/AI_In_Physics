import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import the Live Feeder Machine
from generator import forge_pristine_universe
from interference import FWHM_NORMALIZER

# --- THE UNIVERSAL CONSTANT ---

def worker_init_fn(worker_id):
    """
    The Anti-Cloning Protocol. 
    Guarantees every CPU core generates mathematically unique universes.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

class FabryPerotDataset(Dataset):
    """
    The PyTorch interface that commands the Live Feeder Machine.
    """
    def __init__(self, epoch_size=2000):
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, index):
        # 1. Command the Feeder to forge a universe
        matrix, target_fwhm = forge_pristine_universe()
        
        # --- INPUT NORMALIZATION ---
        matrix_min = np.min(matrix)
        matrix_max = np.max(matrix)
        
        # Failsafe against division by zero in case of a perfectly flat void
        if matrix_max - matrix_min == 0:
            normalized_matrix = matrix - matrix_min
        else:
            normalized_matrix = (matrix - matrix_min) / (matrix_max - matrix_min)
            
        # --- MAXIMUM ABSOLUTE TARGET SCALING ---
        # Compress the true physics safely into the 0.0 to 1.0 realm
        normalized_fwhm = target_fwhm / FWHM_NORMALIZER
        
        # 2. Convert normalized NumPy memory into PyTorch Tensors
        tensor_matrix = torch.tensor(normalized_matrix, dtype=torch.float32).unsqueeze(0)
        tensor_fwhm = torch.tensor(normalized_fwhm, dtype=torch.float32)
        
        return tensor_matrix, tensor_fwhm

def get_live_dataloaders(batch_size=32, train_size=2000, val_size=400):
    """
    The Master Ammunition Belts.
    """
    train_dataset = FabryPerotDataset(epoch_size=train_size)
    val_dataset = FabryPerotDataset(epoch_size=val_size)
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        worker_init_fn=worker_init_fn, 
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        worker_init_fn=worker_init_fn,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    print("Testing the Batch Controller...")
    train_loader, _ = get_live_dataloaders(batch_size=8, train_size=100)
    
    matrix_batch, fwhm_batch = next(iter(train_loader))
    print(f"Batch successfully compiled. Shape: {matrix_batch.shape}")