import torch.nn as nn

from models.decoder import Decoder
from models.encoder import Encoder


class EncoderDecoderPair(nn.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input):
        latent = self.encoder(input)
        out = self.decoder(latent)
        return out
