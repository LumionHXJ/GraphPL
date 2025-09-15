import torch
import torch.nn as nn

from celeba.networks.FeatureExtractorImg import FeatureExtractorImg, DataGeneratorImg
from celeba.networks.FeatureCompressor import LinearFeatureCompressor

from ..constants import styles, modality_dims
DIM_IMG = [64, 128, 256, 512, 512]
class_dim = modality_dims['img']
style_dim = styles['img']

class EncoderImg(nn.Module):
    def __init__(self, flags):
        super(EncoderImg, self).__init__();
        self.feature_extractor = FeatureExtractorImg(3, DIM_IMG)
        self.feature_compressor = LinearFeatureCompressor(DIM_IMG[-1],
                                                          style_dim,
                                                          class_dim)

    def forward(self, x_img):
        h_img = self.feature_extractor(x_img);
        h_img = h_img.view(h_img.shape[0], h_img.shape[1], h_img.shape[2])
        mu_style, logvar_style, mu_content, logvar_content = self.feature_compressor(h_img);
        return mu_style, logvar_style, mu_content, logvar_content, h_img;


class DecoderImg(nn.Module):
    def __init__(self, flags):
        super(DecoderImg, self).__init__()
        self.style_dim = style_dim
        self.feature_generator = nn.Linear(style_dim + class_dim, DIM_IMG[-1], bias=True);
        self.img_generator = DataGeneratorImg(3, DIM_IMG)

    def forward(self, z_style, z_content):
        if self.style_dim > 0:
            z = torch.cat((z_style, z_content), dim=1).squeeze(-1)
        else:
            z = z_content
        img_feat_hat = self.feature_generator(z);
        img_feat_hat = img_feat_hat.view(img_feat_hat.size(0), img_feat_hat.size(1), 1, 1);
        img_hat = self.img_generator(img_feat_hat)
        return img_hat, torch.tensor(0.75).to(z.device);
