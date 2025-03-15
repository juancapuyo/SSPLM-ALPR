import torch
import torch.nn as nn

class LicensePlateDetector(nn.Module):
    def __init__(self):
        super(LicensePlateDetector, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 1280x720 -> 640x360

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 640x360 -> 320x180

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 320x180 -> 160x90

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 160x90 -> 80x45

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 80x45 -> 40x22

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, ceil_mode=True),  # Keeps 40x22 size

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Adaptive pooling to reduce to 1x1 feature map (batch, 1024, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer to predict (x, y, width, height)
        self.fc = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.conv_layers(x)  # Shape: (batch, 1024, 40, 22)
        x = self.global_pool(x)  # Shape: (batch, 1024, 1, 1)
        x = torch.flatten(x, 1)  # Shape: (batch, 1024)
        x = self.fc(x)  # Shape: (batch, 4) -> (x, y, w, h)
        return x