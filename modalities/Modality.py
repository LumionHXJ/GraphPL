
from abc import ABC, abstractmethod
import os
import glog as logger
import torch
import torch.distributions as dist

class Modality(ABC):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name):
        self.data_size = torch.Size((48,));
        self.name = name;
        self.encoder = enc;
        self.decoder = dec;
        self.class_dim = class_dim;
        self.style_dim = style_dim;
        self.likelihood_name = lhood_name;
        self.likelihood = self.get_likelihood(lhood_name);
        num_params = sum(p.numel() for p in enc.parameters() if p.requires_grad) + \
            sum(p.numel() for p in dec.parameters() if p.requires_grad)
        print(name, num_params)

    def get_likelihood(self, name):
        if name == 'laplace':
            pz = dist.Laplace;
        elif name == 'bernoulli':
            pz = dist.Bernoulli;
        elif name == 'normal':
            pz = dist.Normal;
        elif name == 'categorical':
            pz = dist.OneHotCategorical;
        elif issubclass(name, dist.Distribution):
            pz = name
        else:
            logger.info('likelihood not implemented')
            pz = None;
        return pz;


    def save_data(self, d, fn, args):
        raise NotImplementedError()

    def plot_data(self, d):
        raise NotImplementedError()

    def calc_log_prob(self, out_dist, target, norm_value):
        log_prob = out_dist.log_prob(target).sum();
        mean_val_logprob = log_prob/norm_value;
        return mean_val_logprob;


    def save_networks(self, dir_checkpoints):
        torch.save(self.encoder.state_dict(), os.path.join(dir_checkpoints,
                                                           'enc_' + self.name))
        torch.save(self.decoder.state_dict(), os.path.join(dir_checkpoints,
                                                           'dec_' + self.name))

class BernoulliwithFocal(Modality):
    def __init__(self, name, enc, dec, class_dim, style_dim, lhood_name, gamma=2.0):
        super().__init__(name, enc, dec, class_dim, style_dim, lhood_name)
        self.gamma = gamma
    def calc_log_prob(self, out_dist, target, norm_value):
        log_prob = out_dist.log_prob(target) # N, D (typically output by bernoulli)
        pt = torch.where(target>0.5, out_dist.mean, 1-out_dist.mean)
        log_prob = ((1 - pt)**self.gamma) * log_prob
        mean_val_logprob = log_prob.sum()/norm_value
        return mean_val_logprob
    
class MaskedLaplace(Modality):
    def calc_log_prob(self, out_dist, target, norm_value):
        log_prob = out_dist.log_prob(target) * (target > 0)
        mean_val_logprob = log_prob.sum()/norm_value;
        return mean_val_logprob;