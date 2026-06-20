from torch import nn
from torchinfo import summary
import torchvision.models as models
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # grayscale input
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 22 * 22, 128)  # see size note below
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))   # 48->46
        x = torch.relu(self.conv2(x))   # 46->44
        x = self.pool(x)                # 44->22
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout(0.5),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Dropout(0.5),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(in_features=32 * 12 * 12, out_features=2) 
#         )

#     def forward(self, x):
#         x = self.conv_block(x)
#         x = self.classifier(x)
#         return x
    
# train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
#               '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied'
#               ]
# model = Model(train_data)
# summary(model, input_size=(1, 1, 48, 48))
