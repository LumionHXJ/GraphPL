import torch.nn as nn
from celeba.networks.ResidualBlocks import ResidualBlock2dConv, ResidualBlock2dTransposeConv

class FeatureExtractorImg(nn.Module):
    def __init__(self, image_channels, dim_img):
        super(FeatureExtractorImg, self).__init__();
        self.conv1 = nn.Conv2d(image_channels, dim_img[0], kernel_size=3,
                               stride=2, padding=1, dilation=1, bias=False)
        self.resblock1 = ResidualBlock2dConv(dim_img[0], dim_img[1], kernelsize=4, stride=2, padding=1)
        self.resblock2 = ResidualBlock2dConv(dim_img[1], dim_img[2], kernelsize=4, stride=2, padding=1)
        self.resblock3 = ResidualBlock2dConv(dim_img[2], dim_img[3], kernelsize=4, stride=2, padding=1)
        self.resblock4 = ResidualBlock2dConv(dim_img[3], dim_img[4], kernelsize=4, stride=2, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out);
        out = self.resblock2(out);
        out = self.resblock3(out);
        out = self.resblock4(out);
        return out

class DataGeneratorImg(nn.Module):
    def __init__(self, image_channels, dim_img):
        super(DataGeneratorImg, self).__init__()
        self.resblock1 = ResidualBlock2dTransposeConv(dim_img[4], dim_img[3], kernelsize=4, stride=1, padding=0);
        self.resblock2 = ResidualBlock2dTransposeConv(dim_img[3], dim_img[2], kernelsize=4, stride=2, padding=1);
        self.resblock3 = ResidualBlock2dTransposeConv(dim_img[2], dim_img[1], kernelsize=4, stride=2, padding=1);
        self.resblock4 = ResidualBlock2dTransposeConv(dim_img[1], dim_img[0], kernelsize=4, stride=2, padding=1);
        self.conv = nn.ConvTranspose2d(dim_img[0], image_channels, kernel_size=3,
                                       stride=2, padding=1, dilation=1, output_padding=1);

    def forward(self, feats):
        d = self.resblock1(feats);
        d = self.resblock2(d);
        d = self.resblock3(d);
        d = self.resblock4(d);
        d = self.conv(d)
        return d;