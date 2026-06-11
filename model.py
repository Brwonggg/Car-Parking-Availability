import torch
from torch import nn

class Model(nn):
    def __init__(self, x):
        super().__init__
        self.conv_block = nn.Sequential(
                            nn.Conv2d(in_channels=1 , out_channels=1 , kernel_size=3, stride=1, padding=1),
                            nn.ReLU,
                            nn.Dropout(0.25),
                            nn.Conv2d(in_channels=1 , out_channels=1 , kernel_size=3, stride=1, padding=1),
                            nn.ReLU,
                            nn.Dropout(0.5),
                            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(in_features=, out_features=)

        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x