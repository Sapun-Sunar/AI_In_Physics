import torch
import torch.nn as nn

class NoisePredictor(nn.Module):
    def __init__ (self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,2)
        )

    def forward(self, x_t, t):
        t_normalized = t.view(-1,1).float()/100.0
        fused_input = torch.cat((x_t,t_normalized),dim=1)
        return self.network(fused_input)
    
model = NoisePredictor()