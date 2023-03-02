import torch.nn as nn
import timm


class ResNetModel(nn.Module):
    """
    Model Class for ResNet Models
    """
    def __init__(self, num_classes=5, model_name='resnet18', pretrained=True):
        super(ResNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x = self.model(x)
        return x