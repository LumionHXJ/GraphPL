
import numpy as np
from scipy.special import logsumexp
from itertools import cycle
import math

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils import utils
from utils.likelihood import get_latent_samples
from utils.likelihood import log_probs_estimate

from divergence_measures.mm_div import alpha_poe
from divergence_measures.mm_div import poe
from utils.meter import Meter
from utils.BaseExperiment import BaseExperiment_impute

LOG2PI = float(np.log(2.0 * math.pi))

# at the moment: only marginals and joint
def calc_log_likelihood_batch(exp, latents, imputed_mods, batch, c_ind, num_imp_samples=15):
    flags = exp.flags;
    model = exp.model;
    mods = exp.modalities;
    mod_sample = list(mods.keys())[0]

    # preparing decoder forward
    if exp.flags.impute_method == 'graph':
        styles = model.get_random_style_dists(batch[mod_sample].shape[0] * num_imp_samples)
        l_dec = {'content': dict(),
                 'style': dict()}
        for k in styles.keys():
            if model.styles[k] > 0:
                l_dec['style'][k] = model.reparameterize(styles[k][0], styles[k][1])
            else:
                l_dec['style'][k] = None
        
        # content
        latents = {
            'joint': {m_key: [mu.repeat(num_imp_samples, 1), logvar.repeat(num_imp_samples, 1)] for m_key, (mu, logvar) in latents['joint'].items()},
            'single': {m_key: [mu.repeat(num_imp_samples, 1), logvar.repeat(num_imp_samples, 1)] for m_key, (mu, logvar) in latents['single'].items()},
        }
        for mod in imputed_mods:
            mid = model.mod2id(mod)
            content_rep = model.gnn_forward(latents, mid)
            l_dec['content'][mod] = content_rep
    else:
        styles = model.get_random_style_dists(batch[mod_sample].shape[0])
        subset_names = latents['subsets_in_fusion']
        longest_subset = subset_names[utils.longest_ind(subset_names)]
        s_dist = latents['subsets'][longest_subset]
        n_total_samples = s_dist[0].shape[0]*num_imp_samples
        l_subset = {'content': s_dist, 'style': styles}
        mod_names = mods.keys()
        l = get_latent_samples(model, l_subset, num_imp_samples, mod_names)
        # preparing decoder inputs
        l_dec = {'content': l['content']['z'].reshape(n_total_samples, -1), # NxBs, L 
                'style': dict()};
        for m, m_key in enumerate(l['style'].keys()):
            l_dec['style'][m_key] = l['style'][m_key]['z'].reshape(n_total_samples, -1)

    # gathering target for teacher forcing: p(X|z) = p(x0|z)p(x1|x0,z)...
    target_sequence = dict()
    for m_key in imputed_mods:
        if hasattr(model.decoders[m_key], 'teacher_forcing'):
            target_sequence[m_key] = batch[m_key].repeat(num_imp_samples, *(1,)*(len(batch[m_key].shape)-1))
    # decoder forwarding
    gen = model.generate_sufficient_statistics_from_latents(l_dec, imputed_mods, target_sequence) # missing modals
    
    ll = dict()
    # ! only considering unobserved modalities: no joint anymore
    for k, m_key in enumerate(imputed_mods):
        ll_mod = log_probs_estimate(flags,
                                       num_imp_samples,
                                       gen[m_key],
                                       batch[m_key])
        ll[m_key] = ll_mod
    return ll;

def estimate_likelihoods_all(exp: BaseExperiment_impute):
    model = exp.model;
    mods = exp.modalities;

    lhoods = Meter()
    for c_i in range(exp.flags.client_num):
        subset_used = exp.clients_subsets[c_i]
        d_loader = DataLoader(exp.test_impute_dataset[c_i], 
                              batch_size=64, # TODO nll forward more samples at a time
                              shuffle=False,
                              num_workers=4, 
                              drop_last=False)

        for iteration, batch in enumerate(d_loader):
            batch_all = batch[0]
            batch_l = batch[1]
            observed_mask = batch[2]
            batch_all = {k: Variable(batch_all[k]).to(exp.flags.device) for k in batch_all.keys()}
            batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)            

            latents = model.inference(batch_d, subset_used)
            imputed_mods = []
            for mask, m_key in zip(observed_mask[0], mods):
                if mask == 0:
                    imputed_mods.append(m_key)
            ll_batch = calc_log_likelihood_batch(exp, latents,
                                                 imputed_mods,
                                                 batch_all, c_i)
            lhoods._update(ll_batch, batch_size=batch[2].shape[0])
    lhoods_all = lhoods.get_scalar_dict('global_avg')
    all_results, all_lengths = [], []
    for k in lhoods.meters:
        all_results.append(lhoods.meters[k].avg / exp.modalities[k].data_size.numel())
        all_lengths.append(lhoods.meters[k].total_samples * exp.modalities[k].data_size.numel())
    lhoods_all['avg'] = utils.avg_list(all_results, all_lengths)
    return lhoods_all;
