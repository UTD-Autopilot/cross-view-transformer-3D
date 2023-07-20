import torch.nn as nn
from .....registry import BACKBONES

class ModelModule(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone

    def forward(self, batch):
        return self.backbone(batch)
