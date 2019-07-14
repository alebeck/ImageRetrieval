import torch
import torch.nn as nn

from models.decoder import LowerDecoder, UpperDecoder
from models.encoder import LowerEncoder, UpperEncoder

class Autoencoder(nn.Module):
    encoder_lower: LowerEncoder
    encoder_upper: UpperEncoder
    decoder_lower: LowerDecoder
    decoder_upper: UpperDecoder

    def __init__(self, encoder_lower: LowerEncoder, encoder_upper: UpperEncoder, decoder_lower: LowerDecoder, decoder_upper: UpperDecoder):
        super().__init__()
        self.encoder_lower = encoder_lower
        self.encoder_upper = encoder_upper
        self.decoder_lower = decoder_lower
        self.decoder_upper = decoder_upper

    def forward(self, img):
        # Reduced VAE implementation with mean = latent and std all ones
        latent = self.encode(img)
        if self.training:
            noise = torch.randn(latent.size()).to(latent.device)
            out = self.decode(latent + noise)
        else:
            out = self.decode(latent)

        return out, latent

    def encode(self, img):
        latent = self.encoder_upper(self.encoder_lower(img))
        noise = torch.randn(latent.size()).to(latent.device)
        return latent, noise

    def decode(self, latent):
        return self.decoder_upper(self.decoder_lower(latent))
