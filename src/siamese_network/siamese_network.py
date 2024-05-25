import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SiameseNet(nn.Module):
    def __init__(self, num_filter:list):
        super(SiameseNet, self).__init__()
        self.num_filters = num_filter
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3),
            nn.MaxPool2d(2, stride=2)
        )

        self.lin = nn.Sequential(
                nn.Linear(512*8*8, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 16),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(16, 2)
            )

    def forward(self, inputs):
        x = inputs
        x = self.cnn1(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.lin(x)
        return x
    

