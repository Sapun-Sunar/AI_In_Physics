import torch
import torch.nn as nn

class SpectroscopicBrain(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- BLOCK 1: THE MACRO VISION ---
        # in_channels=1 (Grayscale plasma), out_channels=16
        self.macro_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.macro_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- BLOCK 2: THE MICRO HUNT ---
        self.micro_conv1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.micro_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- BLOCK 3: DEEP EXTRACTION ---
        self.micro_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.micro_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- THE MEASURING TAPE (Preserving Spatial Geometry) ---
        self.flatten = nn.Flatten()
                       
        # --- THE REGRESSION HEAD ---
        self.dense1 = nn.Linear(198400, 128)
        
        # THE HIGH-VOLTAGE ARMOR (Surge Protector)
        self.bn_dense1 = nn.BatchNorm1d(128)
        
        # THE SPATIAL TRANSLATION ENGINE (30% Blindfold)
        self.dropout = nn.Dropout(p=0.3)
        
        self.dense2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)
        
        # The Mathematical Saw
        self.relu = nn.ReLU()
        
        # The Titanium Lock
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        The Thermodynamic flow of the tensor.
        """
        x = self.macro_pool(self.relu(self.macro_conv(x)))
        x = self.micro_pool1(self.relu(self.bn1(self.micro_conv1(x))))
        x = self.micro_pool2(self.relu(self.bn2(self.micro_conv2(x))))
        
        x = self.flatten(x)
        
        # [!] Catch the 198,400-wire tsunami and standardize it BEFORE the ReLU threshold
        x = self.relu(self.bn_dense1(self.dense1(x)))
        
        # Inject the chaos into the flattened spatial map
        x = self.dropout(x)
        x = self.relu(self.dense2(x))
        
        # Final prediction bounded strictly between 0.0 and 1.0
        x = self.sigmoid(self.output_layer(x))
        
        return x

if __name__ == "__main__":
    print("Forging the PyTorch Architecture...")
    
    # Instantiate the weapon
    leviathan_mind = SpectroscopicBrain()
    
    # Generate a dummy tensor to test the forward pass (Batch=1, Channels=1, H=250, W=400)
    dummy_universe = torch.randn(1, 1, 250, 400)
    
    # Fire the network
    predicted_fwhm = leviathan_mind(dummy_universe)
    
    print(f"Network compiled successfully. Output shape: {predicted_fwhm.shape}")