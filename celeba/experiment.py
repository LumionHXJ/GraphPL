

import os 
import random
import numpy as np 

import PIL.Image as Image
from PIL import ImageFont 
import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import average_precision_score

from modalities.CelebaImg import Img
from modalities.CelebaText import Text
from celeba.CelebADataset import CelebaDataset
from celeba.networks.ConvNetworkImgClfCelebA import ClfImg as ClfImg
from celeba.networks.ConvNetworkTextClfCelebA import ClfText as ClfText
from celeba.networks.ConvNetworkBinaryClfCelebA import ClfImg as ClfBinary

from celeba.networks.ConvNetworksImgCelebA import EncoderImg, DecoderImg
from celeba.networks.ConvNetworksBinaryCelebA import EncoderImgBinary, DecoderImgBinary
from celeba.networks.ConvNetworksTextCelebA import EncoderText, DecoderText

from utils.BaseExperiment import BaseExperiment_impute
from celeba.constants import indices, modality_dims, styles, style_weights

from .constants import LABELS, IMG_LABELS, CANNY_LABELS, SEMANTIC_LABELS, TEXT_LABELS


class CelebaExperiment_impute(BaseExperiment_impute):
    dataset_name = 'celeba'
    plot_img_size = torch.Size((3, 64, 64))
    labels  = LABELS
    mod_labels = {'img': IMG_LABELS, 'text': TEXT_LABELS, 'canny': CANNY_LABELS, 'seg': SEMANTIC_LABELS}
    def __init__(self, flags, alphabet):
        self.font = ImageFont.truetype('FreeSerif.ttf', 38)
        super().__init__(flags, alphabet)
        self.eval_metric = average_precision_score

    def get_mod_dimensions(self):
        print(modality_dims)
        return modality_dims
    
    def set_model_attr(self):
        self.model.indices = indices
        self.model.styles = styles
        super().set_model_attr()
        return 
    
    def get_transform_celeba(self):        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(self.flags.crop_size_img),
            transforms.ToPILImage(),
            transforms.Resize(size=(self.flags.img_size, self.flags.img_size),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        return transform


    def set_dataset(self):
        transform = self.get_transform_celeba();
        d_train = CelebaDataset(self.flags, self.alphabet, partition=0, transform=transform)
        d_eval = CelebaDataset(self.flags, self.alphabet, partition=1, transform=transform)
        self.dataset_train = d_train;
        self.dataset_test = d_eval;

    def get_modality(self, modality_name):
        if modality_name=='img':
            mod = Img(modality_name, 
                      EncoderImg(self.flags),
                      DecoderImg(self.flags),
                   (3, 64, 64),
                   style_dim=styles[modality_name]);
        elif modality_name=='canny':
            mod = Img(modality_name,
                      EncoderImgBinary(modality_dims[modality_name], styles[modality_name]),
                   DecoderImgBinary(modality_dims[modality_name], styles[modality_name]),
                   (1, 64, 64),
                   style_dim=styles[modality_name]);
        elif modality_name=='seg':
            mod = Img(modality_name,
                      EncoderImgBinary(modality_dims[modality_name], styles[modality_name]),
                   DecoderImgBinary(modality_dims[modality_name], styles[modality_name]),
                   (1, 64, 64),
                   style_dim=styles[modality_name]);
        elif modality_name=='text':
            mod = Text(EncoderText(self.flags),
                    DecoderText(self.flags),
                    self.flags.len_sequence,
                    self.alphabet,
                    torch.Size((3,64,64)),
                    self.font,
                    style_dim=styles[modality_name]);
        else:
            raise KeyError()
        return mod

    def get_test_samples(self, num_images=30):
        dataset = self.dataset_test
        samples = []
        for i in range(num_images):
            c_ind = random.randint(0, self.flags.client_num-1)
            dataset = self.test_impute_dataset[c_ind]
            n_test = len(dataset)
            ix = random.randint(0, n_test-1)
            sample, target, observed_mask = dataset[ix]
            
            for k, key in enumerate(sample):
                sample[key] = sample[key]
            samples.append([ix, sample, observed_mask, c_ind])
        return samples

    def set_clfs(self):
        model_clf_m1 = None
        model_clf_m2 = None
        model_clf_m3 = None
        model_clf_m4 = None
        if self.flags.use_clf:
            model_clf_m1 = ClfImg(self.flags);
            model_clf_m1.load_state_dict(torch.load(os.path.join(self.flags.dir_clf, 'trained_clfs_celeba/clf_m1')))
            model_clf_m1 = model_clf_m1.eval()

            model_clf_m2 = ClfText(self.flags);
            model_clf_m2.load_state_dict(torch.load(os.path.join(self.flags.dir_clf, 'trained_clfs_celeba/clf_m2')))
            model_clf_m2 = model_clf_m2.eval()

            model_clf_m3 = ClfBinary(len(self.labels));
            model_clf_m3.load_state_dict(torch.load(os.path.join(self.flags.dir_clf, 'trained_clfs_celeba/clf_m3')))
            model_clf_m3 = model_clf_m3.eval()

            model_clf_m4 = ClfBinary(len(self.labels));
            model_clf_m4.load_state_dict(torch.load(os.path.join(self.flags.dir_clf, 'trained_clfs_celeba/clf_m4')))
            model_clf_m4 = model_clf_m4.eval()

        clfs = {'img': model_clf_m1,
                'text': model_clf_m2,
                'canny': model_clf_m3,
                'seg': model_clf_m4}
        return clfs;

    def set_rec_weights(self):
        rec_weights = dict()
        ref_mod_d_size = self.modalities['img'].data_size.numel()/3;
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key];
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size/numel_mod)
        print(rec_weights)
        return rec_weights

    def set_style_weights(self):
        print(style_weights)
        return style_weights;
    
    def get_prediction_from_attr(self, values):
        return values.ravel();

    def get_prediction_from_attr_random(self, values, index=None):
        return values[:,index] > 0.5;

    def eval_label(self, values, labels, index):
        pred = values[:,index];
        gt = labels[:,index];
        try:
            ap = self.eval_metric(gt, pred) 
        except ValueError:
            raise ValueError()
        return ap

    def filter_eval_result(self, ap_dict, mod):
        return {k:v for k,v in ap_dict.items() if k in self.mod_labels[mod]}