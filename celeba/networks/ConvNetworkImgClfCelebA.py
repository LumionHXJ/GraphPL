import torch
import torch.nn as nn

class ResidualBlock2dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample, a=1, b=1):
        super(ResidualBlock2dConv, self).__init__();
        self.conv1 = nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False)
        self.dropout1 = nn.Dropout2d(p=0.5, inplace=False)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_in)
        self.conv2 = nn.Conv2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.dropout2 = nn.Dropout2d(p=0.5, inplace=False)
        self.downsample = downsample
        self.a = a;
        self.b = b;

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
        out = self.a*residual + self.b*out;
        return out


def make_res_block_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation, a_val=2.0, b_val=0.3):
    downsample = None;
    if (stride != 2) or (in_channels != out_channels):
        downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                             kernel_size=kernelsize,
                                             padding=padding,
                                             stride=stride,
                                             dilation=dilation),
                                   nn.BatchNorm2d(out_channels))
    layers = [];
    layers.append(ResidualBlock2dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample,a=a_val, b=b_val))
    return nn.Sequential(*layers)


class FeatureExtractorImg(nn.Module):
    def __init__(self, args, a, b):
        super(FeatureExtractorImg, self).__init__();
        self.args = args;
        self.a = a;
        self.b = b;
        self.conv1 = nn.Conv2d(3, 128,
                              kernel_size=3,
                              stride=2,
                              padding=2,
                              dilation=1,
                              bias=False)
        self.resblock1 = make_res_block_feature_extractor(128, 2 * 128, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=a, b_val=b)
        self.resblock2 = make_res_block_feature_extractor(2 * 128, 3 * 128, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock3 = make_res_block_feature_extractor(3 * 128, 4 * 128, kernelsize=4, stride=2,
                                                          padding=1, dilation=1, a_val=self.a, b_val=self.b)
        self.resblock4 = make_res_block_feature_extractor(4 * 128, 5 * 128, kernelsize=4, stride=2,
                                                          padding=0, dilation=1, a_val=self.a, b_val=self.b)

    def forward(self, x):
        out = self.conv1(x)
        out = self.resblock1(out);
        out = self.resblock2(out);
        out = self.resblock3(out);
        out = self.resblock4(out);
        return out


class ClfImg(nn.Module):
    def __init__(self, flags):
        super(ClfImg, self).__init__();
        self.feature_extractor = FeatureExtractorImg(flags, a=2.0, b=0.3);
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=5*128, out_features=40, bias=True);
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
