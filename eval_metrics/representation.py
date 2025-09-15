import sys
import os
from collections import defaultdict
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import utils.utils as utils

def train_clf_lr_all_subsets(exp):
    mm_vae = exp.model;
    mm_vae.eval();

    clf_all = []
    scaler_all = []
    for c_i in range(exp.flags.client_num):
        subset_used = exp.clients_subsets[c_i]
        d_loader = DataLoader(exp.train_impute_dataset[c_i], 
                              batch_size=exp.flags.batch_size,
                              shuffle=True,
                              num_workers=4, drop_last=False);

        bs = exp.flags.batch_size
        n_train_samples = exp.flags.num_training_samples_lr
        labels = []
        data_train = []

        for it, batch in enumerate(d_loader):
            batch_all = batch[0]
            batch_l = batch[1]
            observed_mask = batch[2]

            batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)
            batch_d = {k: Variable(batch_d[k]).to(exp.flags.device) for k in batch_d.keys()}

            inferred = mm_vae.inference(batch_d, subset_used)
            enc_mods = inferred['modalities']
            data = torch.Tensor().to(exp.flags.device)
            for m_key in exp.modalities_names:
                if m_key in batch_d:
                    data = torch.cat((data, enc_mods[m_key][0]), dim=1)
                else:
                    '''if exp.flags.impute_method == 'graph':
                        class_embeddings = mm_vae.gnn_forward(inferred, mid=mm_vae.mod2id(m_key))
                        data = torch.cat((data, class_embeddings), dim=1)
                    else:
                        data = torch.cat((data, inferred['joint'][0]), dim=1)'''
            data_train.append(data.cpu().data.numpy())

            labels.append(batch_l.data.numpy())
            if (it+1)*bs>=n_train_samples:
                break
        
        # training linear regression
        labels = np.concatenate(labels, axis=0)[:n_train_samples]
        scaler_client = StandardScaler()
        data_train = np.concatenate(data_train, axis=0)[:n_train_samples]
        data_train = scaler_client.fit_transform(data_train)
        clf_lr = train_clf_lr(exp, data_train, labels)
        clf_all.append(clf_lr)
        scaler_all.append(scaler_client)
    return clf_all, scaler_all
 
def test_clf_lr_all_subsets(epoch, clf_lr, exp):
    mm_vae = exp.model;
    mm_vae.eval();

    clf_lr, scaler_lr = clf_lr

    client_eval = dict()
    test_len = []
    for c_i in range(exp.flags.client_num):
        subset_used = exp.clients_subsets[c_i]
        d_loader = DataLoader(exp.test_impute_dataset[c_i], 
                              batch_size=exp.flags.batch_size,
                              shuffle=False,
                              num_workers=4, 
                              drop_last=False);

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
            enc_mods = inferred['modalities']
            data = torch.Tensor().to(exp.flags.device)
            for m_key in exp.modalities_names:
                if m_key in batch_d:
                    data = torch.cat((data, enc_mods[m_key][0]), dim=1)
                else:
                    '''if exp.flags.impute_method == 'graph':
                        class_embeddings = mm_vae.gnn_forward(inferred, mid=mm_vae.mod2id(m_key))
                        data = torch.cat((data, class_embeddings), dim=1)
                    else:
                        data = torch.cat((data, inferred['joint'][0]), dim=1)'''
            data = data.cpu().data.numpy()
            data = scaler_lr[c_i].transform(data)
            outputs = predict_latent_representations(exp, clf_lr[c_i], data)
            for label_k in outputs.keys():
                outputs_all[label_k].append(outputs[label_k])
        
        labels_all = torch.cat(labels_all, dim=0)
        labels_all = torch.reshape(labels_all, (labels_all.shape[0], len(exp.labels))).numpy()

        eval_all_labels = {}
        for l, label_str in enumerate(exp.labels):
            gt = labels_all[:, l]
            if gt.all() or not gt.any():
                continue
            outputs_all[label_str] = np.concatenate(outputs_all[label_str])
            eval_label_rep = exp.eval_metric(gt.ravel(), outputs_all[label_str].ravel());
            eval_all_labels[label_str] = eval_label_rep * 100
        if len(eval_all_labels) > 0:
            client_eval[f'c_{c_i}'] = np.mean(list(eval_all_labels.values())) 
            test_len.append(exp.client_test_dataset_len[c_i])
    client_eval['avg'] = utils.avg_list(list(client_eval.values()), test_len)
    return client_eval

def predict_latent_representations(exp, clf_lr, data):
    output_all_labels = dict()
    for l, label_str in enumerate(exp.labels):
        clf_lr_label = clf_lr[label_str];
        if exp.flags.dataset=='celeba' or exp.flags.dataset=='eicu' or exp.flags.dataset=='muse_eicu':
            y_pred_rep = clf_lr_label.predict_proba(data)[:,1]
        else:
            y_pred_rep = clf_lr_label.predict(data);
        output_all_labels[label_str] = y_pred_rep.ravel()

    return output_all_labels;

def train_clf_lr(exp, data, labels):
    labels = np.reshape(labels, (labels.shape[0], len(exp.labels)))
    clf_lr_labels = dict()
    for l, label_str in enumerate(exp.labels):
        gt = labels[:, l];
        clf_lr_s = LogisticRegression(random_state=0, solver='liblinear', C=0.01, 
                                      multi_class='auto', max_iter=100, class_weight='balanced')
        clf_lr_s.fit(data, gt.ravel())
        clf_lr_labels[label_str] = clf_lr_s;
    return clf_lr_labels;
