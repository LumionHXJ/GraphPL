
import torch
import torch.nn as nn
import glog as logger
from utils.utils import Flatten, Unflatten
from ..constants import modality_dims
from .ResidualBlocks import ResidualBlock2dConv, ResidualBlock2dTransposeConv

class_dim = modality_dims['svhn']

class EncoderSVHN(nn.Module):
    def __init__(self, flags, style_dim):
        super(EncoderSVHN, self).__init__()
        logger.info(f"SVHN style dim: {style_dim}, class dim: {class_dim}")
        self.flags = flags
        self.style_dim = style_dim
        self.stem = nn.Sequential(                          # input shape (3, 32, 32)
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),     # -> (16, 32, 32)
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.res_layers = nn.Sequential(
            ResidualBlock2dConv(16, 32, kernelsize=3, stride=2, padding=1),
            ResidualBlock2dConv(32, 64, kernelsize=3, stride=2, padding=1),
            ResidualBlock2dConv(64, 128, kernelsize=3, stride=2, padding=1),
            Flatten(),
            nn.Linear(128 * 4 * 4, style_dim + class_dim),
            nn.BatchNorm1d(style_dim + class_dim),
            nn.ReLU()
        )
        self.class_mu = nn.Linear(style_dim + class_dim, class_dim)
        self.class_logvar = nn.Linear(style_dim + class_dim, class_dim)

        if self.style_dim > 0:
            self.style_mu = nn.Linear(style_dim + class_dim, style_dim)
            self.style_logvar = nn.Linear(style_dim + class_dim, style_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_layers(x)

        if self.style_dim > 0:
            return self.style_mu(x), self.style_logvar(x), self.class_mu(x), self.class_logvar(x)
        else:
            return None, None, self.class_mu(x), self.class_logvar(x)

class DecoderSVHN(nn.Module):
    def __init__(self, flags, style_dim):
        super(DecoderSVHN, self).__init__()
        self.flags = flags
        self.style_dim = style_dim
        # Decoder network with more layers and parameters
        self.decoder = nn.Sequential(
            nn.Linear(style_dim + class_dim, 128 * 4 * 4),
            nn.BatchNorm1d(128 * 4 * 4),
            nn.ReLU(),
            Unflatten((128, 4, 4)),                              # -> (128, 2, 2)
            ResidualBlock2dTransposeConv(128, 64, kernelsize=3, stride=2, padding=1),  # -> (64, 8, 8)
            ResidualBlock2dTransposeConv(64, 32, kernelsize=3, stride=2, padding=1), 
            ResidualBlock2dTransposeConv(32, 16, kernelsize=3, stride=2, padding=1),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1),    # -> (3, 32, 32)
        )

    def forward(self, style_latent_space, class_latent_space):
        if self.style_dim > 0:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self.decoder(z)
        return x_hat, torch.tensor(0.75).to(z.device)  # NOTE: consider learning scale param, too