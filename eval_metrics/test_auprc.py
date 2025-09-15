import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import glog as logger
from utils.meter import Meter
import utils.utils as utils
from sklearn.metrics import average_precision_score
from collections import defaultdict
import numpy as np

# top 50% data
task_num = {'diagnosis': 13, 'treatment': 58, 'medication': 34}

def test_prc_all(epoch, exp):
    mm_vae = exp.model;

    gen_perf_all = []
    outputs_all = defaultdict(list)
    labels_all = defaultdict(list)
    for c_i in range(exp.flags.client_num):
        subset_used = exp.clients_subsets[c_i]
        gen_perf = Meter()

        d_loader = DataLoader(exp.test_impute_dataset[c_i], 
                              batch_size=exp.flags.batch_size,
                              shuffle=False,
                              num_workers=4, 
                              drop_last=False)

        for iteration, batch in enumerate(d_loader):
            batch_all = batch[0]
            batch_l = batch[1]
            observed_mask = batch[2]

            batch_all = {k: Variable(batch_all[k]).to(exp.flags.device) for k in batch_all.keys()}
            batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)
            imputed_mods = set(batch_all.keys()) - set(batch_d.keys())
            if exp.flags.dataset == 'muse_eicu':
                imputed_mods = imputed_mods.intersection(['diagnosis', 'treatment', 'medication'])

            with torch.no_grad():
                inferred = mm_vae.inference(batch_d, subset_used)
                if exp.flags.impute_method == 'graph':
                    output = mm_vae.cond_generation(inferred)
                else:
                    subset_names = inferred['subsets_in_fusion']
                    longest_subset = subset_names[utils.longest_ind(subset_names)]
                    output = mm_vae.cond_generation({longest_subset: inferred['subsets'][longest_subset]})[longest_subset]
                for mod in imputed_mods:
                    outputs_all[mod].append(output[mod])
                    labels_all[mod].append(batch_all[mod])        
        
    result = defaultdict(float)
    for mod in outputs_all.keys():
        output: np.ndarray = torch.cat(outputs_all[mod], dim=0).cpu().numpy()
        labels = torch.cat(labels_all[mod], dim=0).cpu().numpy()
        top_filter = labels.mean(axis=0) >= 0.1
        auprc = average_precision_score(labels[:, top_filter], output[:, top_filter]) * 100
        result[mod] = auprc

    return result
    