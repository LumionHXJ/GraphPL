import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from utils.BaseExperiment import BaseExperiment_impute
from muse_eicu.eicu_dataset import eICUDataset
from modalities.Modality import Modality, BernoulliwithFocal, MaskedLaplace
from .networks.eicu_coder import EICUEncoder, EICUDecoder
from muse_eicu.constants import input_dim, ffn_layers, lhoods, output_fn, modality_dims
class eicuExperiment_impute(BaseExperiment_impute):
    dataset_name = 'eicu'
    plot_img_size = torch.Size((3, 64, 64))
    labels = ['mortality']

    def __init__(self, flags, alphabet):
        super().__init__(flags, alphabet)
        self.eval_metric = average_precision_score     

    def set_dataset(self):
        d_train = eICUDataset(self.flags.dir_data)
        d_eval = d_train
        self.dataset_train = d_train;
        self.dataset_test = d_eval;

    def get_mod_dimensions(self):
        return modality_dims
    
    def set_model_attr(self):
        self.model.indices = {mod: i for i, mod in enumerate(self.modalities_names)}
        self.model.styles = {mod: 0 for i, mod in enumerate(self.modalities_names)}
        super().set_model_attr()
        return 

    def get_modality(self, modality_name):
        if modality_name in ['diagnosis', 'treatment', 'medication']:
            mod = BernoulliwithFocal(modality_name, 
                        EICUEncoder(input_dim[modality_name], ffn_layers[modality_name], modality_dims[modality_name]),
                        EICUDecoder(input_dim[modality_name], ffn_layers[modality_name], 
                                    output_fn[modality_name], modality_dims[modality_name]),
                        modality_dims[modality_name],
                        modality_dims[modality_name],
                        lhood_name=lhoods[modality_name])
        elif modality_name == 'lab': 
            mod = MaskedLaplace(modality_name, 
                        EICUEncoder(input_dim[modality_name], ffn_layers[modality_name], modality_dims[modality_name]),
                        EICUDecoder(input_dim[modality_name], ffn_layers[modality_name], 
                                    output_fn[modality_name], modality_dims[modality_name]),
                        modality_dims[modality_name],
                        modality_dims[modality_name],
                        lhood_name=lhoods[modality_name])
        else:
            mod = Modality(modality_name, 
                        EICUEncoder(input_dim[modality_name], ffn_layers[modality_name], modality_dims[modality_name]),
                        EICUDecoder(input_dim[modality_name], ffn_layers[modality_name], 
                                    output_fn[modality_name], modality_dims[modality_name]),
                        modality_dims[modality_name],
                        modality_dims[modality_name],
                        lhood_name=lhoods[modality_name])
        return mod

    def get_test_samples(self, num_images=10):
        return None

    def set_clfs(self):
        return None

    def get_prediction_from_attr(self, attr, index=None):
        raise NotImplementedError()
        
    def eval_label(self, values, labels, index):
        raise NotImplementedError()

    def set_rec_weights(self):
        rec_weights = dict()
        for k, m_key in enumerate(self.modalities.keys()):
            rec_weights[m_key] = 1.0/len(self.modalities.keys())
        return rec_weights

    def set_style_weights(self):
        weights = {m: self.flags.beta_style for m in self.modalities.keys()}
        return weights

    def get_prediction_from_attr_random(self, values, index=None):
        raise NotImplementedError
