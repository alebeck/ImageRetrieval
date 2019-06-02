import torch.nn as nn

from models.decoder import Decoder
from models.encoder import LowerEncoder, UpperEncoder


class Autoencoder(nn.Module):
    encoder_lower: LowerEncoder
    encoder_upper: UpperEncoder
    decoder: Decoder

    def __init__(self, encoder_lower: LowerEncoder, encoder_upper: UpperEncoder, decoder: Decoder):
        super().__init__()
        self.encoder_lower = encoder_lower
        self.encoder_upper = encoder_upper
        self.decoder = decoder

    def forward(self, img):
        latent = self.encode(img)
        out = self.decoder(latent)
        return out

    def encode(self, img):
        return self.encoder_upper(self.encoder_lower(img))

    def decode(self, latent):
        return self.decoder(latent)
