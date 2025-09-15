import torch
import torch.nn as nn



# Residual block
class ResidualBlock1dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding):
        super(ResidualBlock1dConv, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in);
        self.conv1 = nn.Conv1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_out);
        self.conv2 = nn.Conv1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, bias=False)
        self.downsample = nn.Sequential()
        if stride != 1 or channels_in != channels_out:
            self.downsample = nn.Sequential(
                nn.Conv1d(channels_in, channels_out, kernel_size=kernelsize, padding=padding, stride=stride, bias=False),
                nn.BatchNorm1d(channels_out)
            )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.alpha * residual + out
        out = self.relu(out)
        return out


class ResidualBlock1dTransposeConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding):
        super(ResidualBlock1dTransposeConv, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in);
        self.conv1 = nn.ConvTranspose1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_out);
        self.conv2 = nn.ConvTranspose1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, bias=False)
        self.upsample = nn.Sequential()
        if stride != 1 or channels_in != channels_out:
            self.upsample = nn.Sequential(
                nn.ConvTranspose1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, bias=False),
                nn.BatchNorm1d(channels_out)
            )
        self.alpha = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        residual = self.upsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.alpha * residual + out
        out = self.relu(out)
        return out


class ResidualBlock2dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding):
        super(ResidualBlock2dConv, self).__init__();
        self.conv1 = nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, bias=False)
        self.downsample = nn.Sequential()
        if stride != 1 or channels_in != channels_out:
            self.downsample = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(channels_out)
            )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.alpha * residual + out
        out = self.relu(out)
        return out


class ResidualBlock2dTransposeConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding):
        super(ResidualBlock2dTransposeConv, self).__init__();
        self.conv1 = nn.ConvTranspose2d(channels_in, channels_in, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_in)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, bias=False)
        self.upsample = nn.Sequential()
        if stride != 1 or channels_in != channels_out:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(channels_out)
            )
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = self.upsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.alpha * residual + out
        out = self.relu(out)
        return out