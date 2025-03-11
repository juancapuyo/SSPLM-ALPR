import torch as nn

class YOLOv2Modified(nn.module):

    def __init__(self):
        super(YOLOv2Modified, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 416 -> 208

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 208 -> 104

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 104 -> 52

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 52 -> 26

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 26 -> 13

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 13 -> 13 (padded to keep size)

            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(1024, 5, kernel_size=1)  # 5 output channels
        )

    def forward(self, x):
        x = self.conv_layers(x)  # Output shape: (batch, 7, 13, 13)
        return x