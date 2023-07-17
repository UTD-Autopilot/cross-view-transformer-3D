import torch.nn as nn
from  

@BACKBONES.register_module
class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_last: int = 64,
        dim_out: int = 32,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_out, 1))

    def forward(self, batch):
        x = self.encoder(batch)
        y = self.decoder(x)
        z = self.to_logits(y)

        return z
