from torch import nn
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, num_unfrozen_layers=24):
        super().__init__()
        self.backbone = models.vgg16(weights='IMAGENET1K_V1')
        
        for i, param in enumerate(self.backbone.features.parameters()):
            if i < num_unfrozen_layers:
                param.requires_grad = False
        
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 2) 
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = self.backbone.classifier(x)
        return x

