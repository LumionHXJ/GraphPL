
from abc import ABC, abstractmethod
import pickle
import random
import glog as logger
import torch
import os
from itertools import chain, combinations
import glob
import glog as logger
from torch.utils.data import DataLoader
from collections import defaultdict, OrderedDict
from torch.optim import Adam
from torch import nn
import numpy as np
from sklearn.model_selection import train_test_split
from utils.GraphMMVae import GraphMMVae
from utils.BaseMMVae import BaseMMVae
from utils.dataset import Client_Dataset, get_observed_mask, noniid, celeba_split

def get_subsets(modalities):
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
    (1,2,3)
    """
    xs = list(modalities) # list(self.modalities.keys())
    # note we return an iterator rather than a list
    subsets_list = chain.from_iterable(combinations(xs, n) for n in
                                        range(len(xs)+1))
    subsets = dict();
    for k, mod_names in enumerate(subsets_list):
        mods = [];
        for l, mod_name in enumerate(sorted(mod_names)):
            mods.append(modalities[mod_name])
        key = '_'.join(sorted(mod_names));
        subsets[key] = mods;
    return subsets;

class BaseExperiment_impute(ABC):
    def __init__(self, flags, alphabet):
        self.flags = flags
        self.alphabet = alphabet
        self.num_modalities = flags.num_mods
        self.set_dataset()
        self.modalities_names = self.dataset_train.modalities_names
        self.model = self.set_model()
     
        self.set_impute_dataset()
        self.set_model_attr()
        self.client_train_dataset_len = np.array([len(self.train_impute_dataset[i]) for i in range(len(self.test_impute_dataset))])
        self.client_test_dataset_len = np.array([len(self.test_impute_dataset[i]) for i in range(len(self.test_impute_dataset))])
        self.test_samples = self.get_test_samples()
                
        self.optimizer_class = self.set_optimizer_class()
        self.mu_dim = flags.class_dim
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()
        self.clfs = self.set_clfs()
        self.clients_subsets = self.get_client_subset(self.subsets, self.observed_mask)
        self.observed_mask = torch.tensor(self.observed_mask, dtype=torch.bool)

        self.pickle_record = {"train": {}, "test": {}, "test_metric": {},
                            "clients_train_len": {c: len(self.train_impute_dataset[c]) for c in range(len(self.test_impute_dataset))},
                            "clients_test_len": {c: len(self.test_impute_dataset[c]) for c in range(len(self.test_impute_dataset))}}
    
    def to(self, device):
        self.model = self.model.to(device)
        for module in self.model.modules():
            if isinstance(module, nn.modules.RNNBase):
                module.flatten_parameters()
        self.observed_mask = self.observed_mask.to(device)
        if self.clfs is not None:
            for m in self.clfs:
                if self.clfs[m] is not None:
                    self.clfs[m] = self.clfs[m].to(device)
    
    def init_dataloader(self, clients_sample):
        self.train_dataloader = OrderedDict()
        self.test_dataloader = OrderedDict()
        for c_i in clients_sample:
            self.train_dataloader[c_i] = DataLoader(self.train_impute_dataset[c_i],
                                                    batch_size=self.flags.batch_size,
                                                    shuffle=True, 
                                                    num_workers=4, 
                                                    drop_last=False,
                                                    persistent_workers=True)
            self.test_dataloader[c_i] = DataLoader(self.test_impute_dataset[c_i], 
                                                   batch_size=self.flags.batch_size,
                                                   shuffle=False,
                                                   num_workers=4, 
                                                   drop_last=False)
    
    @abstractmethod
    def set_dataset(self):
        pass;
    
    @abstractmethod
    def get_test_samples(self):
        pass;

    @abstractmethod
    def get_modality(self, modality_name):
        pass
    
    @abstractmethod
    def set_clfs(self):
        pass

    def set_impute_dataset(self):
        if self.flags.observed_mask:
            pth = self.flags.observed_mask
        else:
            pth = "./data/latent_missing_%s_%s.pk"%(self.flags.dataset, str(self.flags.test_missing_ratio))
        if not os.path.exists(pth):
            observed_mask = get_observed_mask(self.flags.client_num, self.num_modalities, self.flags.test_missing_ratio)
            with open(pth, "wb") as f:
                pickle.dump([observed_mask], f)
                logger.info(f"Load observed mask from {pth}")
        else:
            with open(pth, "rb") as f:
                observed_mask = pickle.load(f)[0]
        assert observed_mask.shape[0]==self.flags.client_num
        self.observed_mask = observed_mask

        f_data = self.get_data
        if self.flags.dataset != 'eicu' and self.flags.dataset != 'muse_eicu':
            train_labels_list = f_data(self.dataset_train)
            test_labels_list = f_data(self.dataset_test)
        if self.flags.dataset=='celeba':
            train_ind = celeba_split(train_labels_list, self.flags.client_num)
            test_ind = celeba_split(test_labels_list, self.flags.client_num)
        elif self.flags.dataset=='eicu':
            with open(os.path.join(self.flags.dir_data, 'hospital_idx.pkl'), 'rb') as f:
                hospital_dict = pickle.load(f)
            hospitals_idx = random.sample(hospital_dict.keys(), self.flags.client_num)
            train_ind, test_ind = [], []
            for h in hospitals_idx:
                # [dead, alive] in hospital dict
                # 1 for dead label
                imbalance_rate = max(1, int(len(hospital_dict[h][1]) / len(hospital_dict[h][0])))
                alive_tr, alive_te = train_test_split(hospital_dict[h][1], test_size=0.2)
                dead_tr, dead_te = train_test_split(hospital_dict[h][0], test_size=0.2)
                train_ind.append(alive_tr+dead_tr*imbalance_rate)
                test_ind.append(alive_te+dead_te)
        elif self.flags.dataset=='muse_eicu':
            with open(os.path.join(self.flags.dir_data, 'hospital.pkl'), 'rb') as f:
                hospital_dict = pickle.load(f)
            sorted_hosp = sorted(hospital_dict.items(), key=lambda x: len(x[1]), reverse=True)
            top_hosp = [item[0] for item in sorted_hosp[:min(int(1.5 * self.flags.client_num), len(sorted_hosp))]]            
            hospitals_idx = random.sample(top_hosp, self.flags.client_num)
            train_ind, test_ind = [], []
            minimal_trainset = np.inf
            for h in hospitals_idx:
                tr, te = train_test_split(hospital_dict[h], test_size=0.2, random_state=self.flags.seed)
                train_ind.append(tr)
                test_ind.append(te)
                minimal_trainset = min(len(tr), minimal_trainset)
            self.trainset_samples = minimal_trainset
        else:
            train_ind, rand_set_all = noniid(train_labels_list, num_users=self.flags.client_num, shard_per_user=self.flags.class_per_user, num_classes=self.num_classes, rand_set_all=[])
            test_ind, _ = noniid(test_labels_list, num_users=self.flags.client_num, shard_per_user=self.flags.class_per_user, num_classes=self.num_classes, rand_set_all=rand_set_all)

        self.train_impute_dataset = []
        self.test_impute_dataset = []
        for c_i in range(self.flags.client_num):
            self.train_impute_dataset.append(Client_Dataset(self.dataset_train, train_ind[c_i], observed_mask[c_i], 'train'))
            self.test_impute_dataset.append(Client_Dataset(self.dataset_test, test_ind[c_i], observed_mask[c_i], 'test',))

    def set_model_attr(self):
        # checking modality index
        for id, mod_name in enumerate(self.modalities.keys()):
            assert self.model.indices[mod_name] == id

    def get_mod_dimensions(self):
        pass

    def get_data(self, dataset):
        if self.flags.dataset=='polymnist':
            dp = list(dataset.file_paths.keys())[0]
            label = [int(dataset.file_paths[dp][index].split(".")[-2]) for index in range(len(dataset))] 
        elif self.flags.dataset=='celeba':
            label = dataset.attributes
        elif self.flags.dataset=='mnistsvhntext':
            label = []
            for i in range(len(dataset)):
                label.append(int(dataset.labels_mnist[dataset.mnist_idx[i]]))

        return label

    def set_model(self):
        mods = {modality_name: self.get_modality(modality_name) for modality_name in self.modalities_names}
        self.modalities = mods
        self.subsets = get_subsets(mods)
        if self.flags.impute_method == 'graph':
            model = GraphMMVae(self.flags, mods, self.get_mod_dimensions())
        else:
            model = BaseMMVae(self.flags, mods, self.subsets, self.get_mod_dimensions())
        
        if self.flags.load_saved:
            result = model.load_state_dict(torch.load(self.flags.checkpoint_pth), strict=False)
            logger.info('Load checkpoint: %s'%self.flags.checkpoint_pth)
            if result.missing_keys:
                logger.info(f"Missing keys in the state_dict: {result.missing_keys}")
            if result.unexpected_keys:
                logger.info(f"Unexpected keys in the state_dict: {result.unexpected_keys}")
        return model

    def set_optimizer_class(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info('num parameters: %.2f M'%(float(total_params)/1e6))
        optimizer_class = Adam
        return optimizer_class

    def get_client_subset(self, subsets, observed_mask):
        subset_all = []
        for c_i in range(self.flags.client_num):
            subset_i = []
            for c_j in range(self.flags.client_num):
                if c_i==c_j:
                    continue
                overlap_inds = np.where((observed_mask[c_i]*observed_mask[c_j])>0)[0]
                if len(overlap_inds)>0:
                    subset_name = '_'.join(sorted([self.modalities_names[i] for i in overlap_inds]))
                    assert subset_name in subsets.keys()
                    subset_i.append(subset_name)
            subset_all.append(sorted(list(set(subset_i))))
        return subset_all
