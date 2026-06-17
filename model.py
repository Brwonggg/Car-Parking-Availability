from torch import nn
from torchinfo import summary

class Model(nn.Module):
    def __init__(self, x):
        super().__init__()
        self.conv_block = nn.Sequential(
                            nn.Conv2d(in_channels=3 , out_channels=8 , kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.MaxPool2d(kernel_size=2),
                            nn.Dropout(0.7),
                            nn.Conv2d(in_channels=8 , out_channels=16 , kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),
                            nn.Dropout(0.7),
                            nn.MaxPool2d(kernel_size=2),
                            
        )

        self.classifier = nn.Sequential(
                            nn.Flatten(),
                            nn.Linear(in_features=16 * 12 * 12, out_features=2)
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
