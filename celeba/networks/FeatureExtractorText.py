import torch.nn as nn
import torch
from celeba.networks.ResidualBlocks import ResidualBlock1dConv, ResidualBlock1dTransposeConv

class FeatureExtractorText(nn.Module):
    def __init__(self, num_features, dim_text):
        super(FeatureExtractorText, self).__init__()
        self.conv1 = nn.Conv1d(num_features, dim_text[0],
                               kernel_size=4, stride=2, padding=1); # 
        self.resblock_1 = ResidualBlock1dConv(dim_text[0], dim_text[1], kernelsize=4, stride=2, padding=1);
        self.resblock_2 = ResidualBlock1dConv(dim_text[1], dim_text[2], kernelsize=4, stride=2, padding=1);
        self.resblock_3 = ResidualBlock1dConv(dim_text[2], dim_text[3], kernelsize=4, stride=2, padding=1);
        self.resblock_4 = ResidualBlock1dConv(dim_text[3], dim_text[4], kernelsize=4, stride=2, padding=1);
        self.resblock_5 = ResidualBlock1dConv(dim_text[4], dim_text[5], kernelsize=4, stride=2, padding=1);
        self.resblock_6 = ResidualBlock1dConv(dim_text[5], dim_text[6], kernelsize=4, stride=2, padding=0);

    def forward(self, x):
        x = x.transpose(-2,-1);
        out = self.conv1(x)
        out = self.resblock_1(out)
        out = self.resblock_2(out)
        out = self.resblock_3(out)
        out = self.resblock_4(out)
        out = self.resblock_5(out)
        out = self.resblock_6(out)
        return out
