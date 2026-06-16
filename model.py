from torch import nn
from torchinfo import summary

class Model(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(in_channels=1 , out_channels=16 , kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Dropout(0.25),
                            nn.Conv2d(in_channels=16 , out_channels=32 , kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(in_features=32 * 24 * 24, out_features=2)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x
    
# train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
#               '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied'
#               ]
# model = Model(train_data)
# summary(model, input_size=(1, 1, 48, 48))
