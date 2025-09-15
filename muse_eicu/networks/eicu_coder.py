import torch
import torch.nn as nn
from ..constants import ffn_layers

class MLPBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(indim, outdim),
             nn.BatchNorm1d(outdim),
             nn.ReLU(), nn.Dropout(0.1)
        )
        self.reslink = indim == outdim
    def forward(self, x):
        if self.reslink:
            return x + self.ffn(x)
        else:
            return self.ffn(x)

class EICUEncoder(nn.Module):
    def __init__(self, input_dim, ffn_layers, class_dim=64):
        super().__init__()
        self.ffn = nn.ModuleList([MLPBlock(input_dim, class_dim * 2)])
        for i in range(1, ffn_layers):
            self.ffn.append(MLPBlock(class_dim * 2, class_dim * 2))
        self.class_mu = nn.Linear(2*class_dim, class_dim)
        self.class_logvar = nn.Linear(2*class_dim, class_dim)

    def forward(self, x):
        # x: [N, 3]
        for ffn in self.ffn:
            x = ffn(x)
        mu = self.class_mu(x)
        logvar = self.class_logvar(x)
        return None, None, mu, logvar

class EICUDecoder(nn.Module):
    def __init__(self, input_dim, ffn_layers, output_fn, class_dim=64):
        super().__init__()
        self.ffn = nn.ModuleList([MLPBlock(class_dim, class_dim * 2)])
        for i in range(1, ffn_layers):
            self.ffn.append(MLPBlock(class_dim * 2, class_dim * 2))
        self.ffn.append(nn.Linear(class_dim * 2, input_dim))
        self.output_fn = output_fn

    def forward(self, z_style, z_content):
        x = z_content
        for ffn in self.ffn:
            x = ffn(x)
        return self.output_fn(x)