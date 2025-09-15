import torch
import torch.nn as nn
import glog as logger
from utils.utils import Flatten, Unflatten
from ..constants import modality_dims

dataSize = torch.Size([1, 28, 28])
class_dim = modality_dims['mnist']

class EncoderImg(nn.Module):
    def __init__(self, flags, style_dim):
        super(EncoderImg, self).__init__()
        logger.info(f"MNIST style dim: {style_dim}, class dim: {class_dim}")
        self.flags = flags
        self.style_dim = style_dim
        self.shared_encoder = nn.Sequential(                          # input shape (1, 28, 28)
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),      # -> (8, 14, 14)
            nn.BatchNorm2d(16),                                        # Batch Normalization
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),     # -> (16, 7, 7)
            nn.BatchNorm2d(32),                                       # Batch Normalization
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (32, 4, 4)
            nn.BatchNorm2d(64),                                       # Batch Normalization
            nn.ReLU(),
            Flatten(),                                             # -> (512)
            nn.Linear(1024, style_dim + class_dim),              # -> (ndim_private + ndim_shared)
            nn.BatchNorm1d(style_dim + class_dim),              # Batch Normalization for fully connected layer
            nn.ReLU(),
        )

        # Content branch
        self.class_mu = nn.Linear(style_dim + class_dim, class_dim)
        self.class_logvar = nn.Linear(style_dim + class_dim, class_dim)

        # Optional style branch
        if style_dim > 0:
            self.style_mu = nn.Linear(style_dim + class_dim, style_dim)
            self.style_logvar = nn.Linear(style_dim + class_dim, style_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        if self.style_dim > 0:
            return self.style_mu(h), self.style_logvar(h), self.class_mu(h), \
                   self.class_logvar(h)
        else:
            return None, None, self.class_mu(h), self.class_logvar(h)


class DecoderImg(nn.Module):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, flags, style_dim):
        super(DecoderImg, self).__init__()
        self.flags = flags
        self.style_dim = style_dim
        self.decoder = nn.Sequential(
            nn.Linear(style_dim + class_dim, 1024),                                # -> (512)
            nn.BatchNorm1d(1024),                                                        # Batch Normalization for fully connected layer
            nn.ReLU(),
            Unflatten((64, 4, 4)),                                                      # -> (32, 4, 4)
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),             # -> (16, 7, 7)
            nn.BatchNorm2d(32),                                                         # Batch Normalization
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (8, 14, 14)
            nn.BatchNorm2d(16),                                                          # Batch Normalization
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (1, 28, 28)
        )

    def forward(self, style_latent_space, class_latent_space):
        if self.style_dim > 0:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self.decoder(z)
        return x_hat, torch.tensor(0.75).to(z.device)  # NOTE: consider learning scale param, too
