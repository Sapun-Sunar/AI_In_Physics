import torch
import torch.nn as nn
import torch.nn.functional as F


class Spectroscopic_Brain(nn.Module):
    def __init__(self):
        super(Spectroscopic_Brain,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((5,5))

        self.fc1 = nn.Linear(256*5*5,512)
        
        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(512,128)

        self.final_output = nn.Linear(128,1)



    def forward(self,x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = self.adaptive_pool(x)

        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        prediction = self.final_output(x)

        return prediction
    


if __name__ == "__main__":
    dummy_input = torch.randn(1,1,245,245)
    model = Spectroscopic_Brain()

    output = model(dummy_input)

    print(f"The predicted width of the abosrption line is: {output}")