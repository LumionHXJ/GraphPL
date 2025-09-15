import torch
import torch.nn as nn

class ResidualBlock2dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample):
        super(ResidualBlock2dConv, self).__init__();
        self.conv1 = nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.Conv2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = residual + out
        return out


def make_res_block_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation):
    downsample = None;
    if (stride != 2) or (in_channels != out_channels):
        downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=kernelsize,
                                             padding=padding,
                                             stride=stride,
                                             dilation=dilation),
                                   nn.BatchNorm2d(out_channels))
    layers = [];
    layers.append(ResidualBlock2dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample))
    return nn.Sequential(*layers)


class FeatureExtractorImg(nn.Module):
    def __init__(self):
        super(FeatureExtractorImg, self).__init__();
        self.conv1 = nn.Conv2d(1, 64,
                              kernel_size=3,
                              stride=2,
                              padding=2,
                              dilation=1,
                              bias=False)
        self.resblock1 = make_res_block_feature_extractor(64, 2 * 64, kernelsize=4, stride=2,
                                                          padding=1, dilation=1)
        self.resblock2 = make_res_block_feature_extractor(2 * 64, 3 * 64, kernelsize=4, stride=2,
                                                          padding=1, dilation=1)
        self.resblock3 = make_res_block_feature_extractor(3 * 64, 4 * 64, kernelsize=4, stride=2,
                                                          padding=1, dilation=1)
        self.resblock4 = make_res_block_feature_extractor(4 * 64, 5 * 64, kernelsize=4, stride=2,
                                                          padding=0, dilation=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out);
        out = self.resblock2(out);
        out = self.resblock3(out);
        out = self.resblock4(out);
        return out


class ClfImg(nn.Module):
    def __init__(self, out_features):
        super(ClfImg, self).__init__();
        self.feature_extractor = FeatureExtractorImg();
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=5*64, out_features=out_features, bias=True);
        self.sigmoid = nn.Sigmoid();

    def forward(self, x_img):
        h = self.feature_extractor(x_img);
        h = self.dropout(h);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h)
        return out;

    def get_activations(self, x_img):
        h = self.feature_extractor(x_img);
        return h;