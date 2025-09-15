

import torch
from torchvision import transforms

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from modalities.Modality import Modality

from utils import utils
from utils.save_samples import write_samples_img_to_file


class Img(Modality):
    def __init__(self, name, enc, dec, data_size, style_dim):
        self.name = name
        self.likelihood_name = 'laplace';
        self.data_size = torch.Size(data_size)
        self.plot_img_size = torch.Size(data_size)
        self.transform_plot = self.get_plot_transform();
        self.gen_quality_eval = True;
        self.file_suffix = '.png';
        self.encoder = enc;
        self.decoder = dec;
        self.likelihood = self.get_likelihood(self.likelihood_name);
        self.style_dim = style_dim
        num_params = sum(p.numel() for p in enc.parameters() if p.requires_grad) + \
            sum(p.numel() for p in dec.parameters() if p.requires_grad)
        print(name, num_params)


    def save_data(self, d, fn, args):
        img_per_row = args['img_per_row'];
        write_samples_img_to_file(d, fn, img_per_row);

 
    def plot_data(self, d):
        out = self.transform_plot(d.squeeze(0).cpu()).unsqueeze(0);
        return out;


    def get_plot_transform(self):
        transf = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize(size=list(self.plot_img_size)[1:],
                                                       interpolation=Image.BICUBIC),
                                     transforms.ToTensor()])
        return transf;
