from torch import nn
from torchinfo import summary
import torchvision.models as models

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        for param in self.backbone.parameters():
            param.requires_grad = False  # freeze pretrained layers
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)  # replace final layer

    def forward(self, x):
        return self.backbone(x)
    
# train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
#               '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied'
#               ]
# model = Model(train_data)
# summary(model, input_size=(1, 1, 48, 48))
