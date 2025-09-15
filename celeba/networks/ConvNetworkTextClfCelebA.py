
import torch.nn as nn

class ResidualBlock1dConv(nn.Module):
    def __init__(self, channels_in, channels_out, kernelsize, stride, padding, dilation, downsample, a=2, b=0.3):
        super(ResidualBlock1dConv, self).__init__()
        self.bn1 = nn.BatchNorm1d(channels_in);
        self.conv1 = nn.Conv1d(channels_in, channels_in, kernel_size=1, stride=1, padding=0)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False);
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(channels_in);
        self.conv2 = nn.Conv1d(channels_in, channels_out, kernel_size=kernelsize, stride=stride, padding=padding, dilation=dilation)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False);
        self.downsample = downsample;
        self.a = a;
        self.b = b;

    def forward(self, x):
        residual = x;
        out = self.bn1(x)
        out = self.relu(out);
        out = self.conv1(out)
        out = self.dropout1(out);
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout2(out);
        if self.downsample:
            residual = self.downsample(x);
        out = self.a*residual + self.b*out
        return out

def make_res_block_encoder_feature_extractor(in_channels, out_channels, kernelsize, stride, padding, dilation, a_val=2.0, b_val=0.3):
    downsample = None;
    if (stride != 1) or (in_channels != out_channels) or dilation != 1:
        downsample = nn.Sequential(nn.Conv1d(in_channels, out_channels,
                                             kernel_size=kernelsize,
                                             stride=stride,
                                             padding=padding,
                                             dilation=dilation),
                                   nn.BatchNorm1d(out_channels))
    layers = []
    layers.append(ResidualBlock1dConv(in_channels, out_channels, kernelsize, stride, padding, dilation, downsample, a=a_val, b=b_val))
    return nn.Sequential(*layers)


class FeatureExtractorText(nn.Module):
    def __init__(self, args, a, b):
        super(FeatureExtractorText, self).__init__()
        self.args = args
        self.a = a
        self.b = b
        self.conv1 = nn.Conv1d(self.args.num_features, 128,
                               kernel_size=4, stride=2, padding=1, dilation=1);
        self.resblock_1 = make_res_block_encoder_feature_extractor(128, 2*128,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_2 = make_res_block_encoder_feature_extractor(2*128, 3*128,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_3 = make_res_block_encoder_feature_extractor(3*128, 4*128,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_4 = make_res_block_encoder_feature_extractor(4*128, 5*128,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_5 = make_res_block_encoder_feature_extractor(5*128, 5*128,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_6 = make_res_block_encoder_feature_extractor(5*128, 5*128,
                                                                   kernelsize=4, stride=2, padding=0, dilation=1);

    def forward(self, x):
        x = x.transpose(-2,-1);
        out = self.conv1(x)
        out = self.resblock_1(out);
        out = self.resblock_2(out);
        out = self.resblock_3(out);
        out = self.resblock_4(out);
        out = self.resblock_5(out);
        out = self.resblock_6(out);
        return out


class ClfText(nn.Module):
    def __init__(self, flags):
        super(ClfText, self).__init__();
        self.args = flags; 
        self.conv1 = nn.Conv1d(self.args.num_features, 128,
                               kernel_size=3, stride=2, padding=1, dilation=1);
        self.resblock_1 = make_res_block_encoder_feature_extractor(128, 2*128,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_2 = make_res_block_encoder_feature_extractor(2*128, 3*128,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_3 = make_res_block_encoder_feature_extractor(3*128, 4*128,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_4 = make_res_block_encoder_feature_extractor(4*128, 5*128,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_5 = make_res_block_encoder_feature_extractor(5*128, 6*128,
                                                                   kernelsize=4, stride=2, padding=1, dilation=1);
        self.resblock_6 = make_res_block_encoder_feature_extractor(6*128, 7*128,
                                                                   kernelsize=4, stride=2, padding=0, dilation=1);
        self.dropout = nn.Dropout(p=0.5, inplace=False);
        self.linear = nn.Linear(in_features=flags.num_layers_text*128, out_features=40, bias=True)
        self.sigmoid = nn.Sigmoid();


    def forward(self, x_text):
        x_text = x_text.transpose(-2,-1);
        out = self.conv1(x_text)
        out = self.resblock_1(out);
        out = self.resblock_2(out);
        out = self.resblock_3(out);
        out = self.resblock_4(out);
        out = self.resblock_5(out);
        out = self.resblock_6(out);
        h = self.dropout(out);
        h = h.view(h.size(0), -1);
        h = self.linear(h);
        out = self.sigmoid(h)
        return out;
