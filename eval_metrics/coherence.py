import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import glog as logger
from collections import defaultdict

import utils.utils as utils
from utils.meter import Meter

def dict2str(out_dict):
    str_out = ''
    for i, k in enumerate(out_dict.keys()):
        str_out += str(k)+':'
        if isinstance(out_dict[k], dict):
            for k_k in out_dict[k].keys():
                str_out += ' %.2f (%s)'%(out_dict[k][k_k], k_k)
        else:
            str_out += '%.2f'%(out_dict[k])
        
        str_out += '; '
    return str_out

def predict_cond_gen_samples(exp, labels, cond_samples):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)));
    clfs = exp.clfs;
    outputs_mods = {}
    for key in clfs.keys():
        if key in cond_samples.keys():
            mod_cond_gen = cond_samples[key];
            mod_clf = clfs[key];
            '''if key == 'text':
                from utils.text import tensor_to_text
                text_sample = tensor_to_text(exp.modalities[key].alphabet, mod_cond_gen)
                text_sample = [''.join(t).translate({ord('*'): None}) for t in text_sample]
                print(text_sample)'''
            attr_hat = mod_clf(mod_cond_gen);
            outputs_mods[key]=attr_hat
        else:
            logger.info(str(key) + 'not existing in cond_gen_samples');
            raise KeyError()
    return outputs_mods;

def calculate_coherence(exp, samples):
    clfs = exp.clfs;
    mods = exp.modalities;
    c_labels = dict();
    num_sample = samples[list(samples.keys())[0]].shape[0]
    
    pred_mods = np.zeros((len(samples.keys()), num_sample, len(exp.labels)))
    for k, m_key in enumerate(samples.keys()):
        mod = mods[m_key];
        clf_mod = clfs[mod.name];
        samples_mod = samples[mod.name];
        attr_mod = clf_mod(samples_mod);
        output_prob_mod = attr_mod.cpu().data.numpy();
        if exp.flags.dataset=='celeba':
            pred_mod = (output_prob_mod>0.5).astype(int)
        else:
            pred_mod = np.argmax(output_prob_mod, axis=1).astype(int)[:,None];
        pred_mods[k] = pred_mod;
    for j, l_key in enumerate(exp.labels):
        coh_mods = np.all(pred_mods[:, :, j] == pred_mods[0, :, j], axis=0)
        coherence = np.sum(coh_mods.astype(int))/float(num_sample);
        c_labels[l_key] = coherence;
    return c_labels;

def test_generation_all(epoch, exp):
    mods = exp.modalities;
    mm_vae = exp.model;

    with torch.no_grad():
        gen_perf_all = {'cond': {'mean': {l_key: [] for l_key in mods.keys()}}}
        for c_i in range(exp.flags.client_num):
            subset_used = exp.clients_subsets[c_i]
            gen_perf = {'cond': {}}

            d_loader = DataLoader(exp.test_impute_dataset[c_i], 
                                  batch_size=exp.flags.batch_size,
                                  shuffle=False,
                                  num_workers=4, drop_last=False);
            
            outputs_all = defaultdict(list)
            labels_all = []
            for iteration, batch in enumerate(d_loader):
                batch_all = batch[0]
                batch_l = batch[1]
                observed_mask = batch[2]
                labels_all.append(batch_l)

                batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)
                batch_d = {k: Variable(batch_d[k]).to(exp.flags.device) for k in batch_d.keys()}

                inferred = mm_vae.inference(batch_d, subset_used)
                if exp.flags.impute_method == 'graph':
                    output = mm_vae.cond_generation(inferred)
                else:
                    subset_names = inferred['subsets_in_fusion']
                    longest_subset = subset_names[utils.longest_ind(subset_names)]
                    output = mm_vae.cond_generation({longest_subset: inferred['subsets'][longest_subset]})[longest_subset]
                clf_outputs = predict_cond_gen_samples(exp, batch_l, output);
                for k in clf_outputs.keys():
                    outputs_all[k].append(clf_outputs[k])
            # ! organize outputs by clients
            labels_all = torch.cat(labels_all, dim=0)
            clf_cg = {}
            for k in outputs_all.keys():
                eval_labels = {}
                outputs_all[k] = torch.cat(outputs_all[k], dim=0)
                for l, label_str in enumerate(exp.labels):
                    score = exp.eval_label(outputs_all[k].cpu().data.numpy(), labels_all,
                                           index=l)
                    eval_labels[label_str] = score
                if hasattr(exp, 'filter_eval_result'): # used in celeba when clf is not confident
                    eval_labels = exp.filter_eval_result(eval_labels, k)
                clf_cg[k] = np.mean(list(eval_labels.values())) * 100
            gen_perf['cond']['mean'] = clf_cg
            logger.info(f'client {c_i}: Impute result {dict2str(gen_perf["cond"]["mean"])}')

            # gather all imputed
            imputed_keys = list(set(batch_all.keys()) - set(batch_d.keys()))
            for l_key in imputed_keys: # m0, m2, m4
                gen_perf_all['cond']['mean'][l_key].append(gen_perf['cond']['mean'][l_key])

    result = {'cond':{'mean':{}}}
    # gather imputed results from different clients
    all_imputed_results, all_imputed_size = [], []
    for j, l_key in enumerate(exp.modalities.keys()):
        test_dataset_len = exp.client_test_dataset_len[exp.observed_mask[:, j].cpu().numpy() == 0]
        if len(test_dataset_len) == 0:
            break
        result['cond']['mean'][l_key] = utils.avg_list(gen_perf_all['cond']['mean'][l_key], test_dataset_len)
        all_imputed_results.extend(gen_perf_all['cond']['mean'][l_key])
        all_imputed_size.extend(test_dataset_len)
    result['cond']['mean']['avg'] = utils.avg_list(all_imputed_results, all_imputed_size)
    
    return result

def test_noise_imputation(epoch, exp):
    mods = exp.modalities;
    mm_vae = exp.model;
    for client in range(exp.flags.client_num):
        with torch.no_grad():
            subset_used = exp.clients_subsets[client]
            d_loader = DataLoader(exp.test_impute_dataset[client], 
                                    batch_size=exp.flags.batch_size,
                                    shuffle=False,
                                    num_workers=4, drop_last=False);
            observed_mask = exp.observed_mask
            unobserved_mods = []
            observed_mods = []
            for i in range(exp.num_modalities):
                if observed_mask[client, i] == 0:
                    unobserved_mods.append(exp.modalities_names[i])
                else:
                    observed_mods.append(exp.modalities_names[i])
            df = pd.DataFrame(columns=['Noise', 'Mod'] + [mod+" Result" for mod in unobserved_mods])
            for noisy_mod in observed_mods:
                for noise in torch.linspace(0, 1, 11):
                    outputs_all = defaultdict(list)
                    labels_all = []
                    for iteration, batch in enumerate(d_loader):
                        batch_all = batch[0]
                        batch_l = batch[1]
                        observed_mask = batch[2]
                        labels_all.append(batch_l)

                        batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)
                        batch_d = {k: Variable(batch_d[k]).to(exp.flags.device) for k in batch_d.keys()}

                        batch_d[noisy_mod] = torch.rand_like(batch_d[noisy_mod]).to(exp.flags.device) * noise + batch_d[noisy_mod] * (1-noise)

                        inferred = mm_vae.inference(batch_d, subset_used)
                        if exp.flags.impute_method == 'graph':
                            output = mm_vae.cond_generation(inferred)
                        else:
                            subset_names = inferred['subsets_in_fusion']
                            longest_subset = subset_names[utils.longest_ind(subset_names)]
                            output = mm_vae.cond_generation({longest_subset: inferred['subsets'][longest_subset]})[longest_subset]
                        clf_outputs = predict_cond_gen_samples(exp, batch_l, output);
                        for k in clf_outputs.keys():
                            outputs_all[k].append(clf_outputs[k])
                    labels_all = torch.cat(labels_all, dim=0)
                    clf_cg = {'Noise': round(float(noise), 2), 'Mod': noisy_mod}
                    for k in list(set(batch_all.keys()) - set(batch_d.keys())): # m1 and m3
                        eval_labels = {}
                        outputs_all[k] = torch.cat(outputs_all[k], dim=0)
                        for l, label_str in enumerate(exp.labels):
                            score = exp.eval_label(outputs_all[k].cpu().data.numpy(), labels_all,
                                                index=l)
                            eval_labels[label_str] = score
                        if hasattr(exp, 'filter_eval_result'): # used in celeba when clf is not confident
                            eval_labels = exp.filter_eval_result(eval_labels, k)
                        clf_cg[k+" Result"] = np.mean(list(eval_labels.values())) * 100
                    df.loc[len(df)] = clf_cg
                    logger.info(clf_cg)
        if exp.flags.impute_method != 'graph':
            print(longest_subset)
        os.makedirs(f'noisy_impute/seed{exp.flags.seed}_c{client}', exist_ok=True)
        df.to_csv(f'noisy_impute/seed{exp.flags.seed}_c{client}/{exp.flags.impute_method}_noise_data.csv', index=False)
    return