import torch.nn as nn
from ...registry import BACKBONES
from .encoder import Encoder
from .decoder import Decoder

@BACKBONES.register_module
class CrossViewTransformerBackbone(nn.Module):
    def __init__(
        self,
        encoder_args, 
        decoder_args,
        ds_factor=None,
        dim_last: int = 64,
        dim_out: int = 32,
    ):
        super().__init__()
        self.encoder = Encoder(**encoder_args)
        self.decoder = Decoder(**decoder_args)

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
