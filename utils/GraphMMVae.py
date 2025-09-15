from abc import ABC, abstractmethod
import copy
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from divergence_measures.kl_div import calc_kl_divergence
from divergence_measures.mm_div import calc_alphaJSD_modalities
from divergence_measures.mm_div import calc_group_divergence_moe
from divergence_measures.mm_div import poe
from utils import utils
from utils.BaseGCN import GroupBlock, compute_class_dim, construct_graph


class GraphMMVae(ABC, nn.Module):
    def __init__(self, flags, modalities, mod_dimensions):
        super(GraphMMVae, self).__init__()
        self.num_modalities = len(modalities.keys());
        self.flags = flags;
        self.modalities = modalities;
        self.mod_dimensions = mod_dimensions
        self.set_fusion_functions();

        encoders = nn.ModuleDict();
        decoders = nn.ModuleDict();
        lhoods = dict();
        for m, m_key in enumerate(modalities.keys()):
            encoders[m_key] = modalities[m_key].encoder;
            decoders[m_key] = modalities[m_key].decoder;
            lhoods[m_key] = modalities[m_key].likelihood;
        self.encoders = encoders;
        self.decoders = decoders;
        self.lhoods = lhoods;

        self.dim_multiples = dict()
        self.class_dim = compute_class_dim(flags.class_dim, *mod_dimensions.values())
        self.num_heads = self.class_dim // flags.class_dim
        for mod in self.modalities.keys():
            assert self.class_dim % mod_dimensions[mod] == 0
            self.dim_multiples[mod] = self.class_dim // mod_dimensions[mod]

        # gcn, observed_mask feed into function
        self.cm_graph_layers = nn.ModuleList()
        for _ in range(flags.gnn_layers):
            self.cm_graph_layers.append(GroupBlock(flags.class_dim, self.num_heads))
        self.modality_embedding = nn.Embedding(self.num_modalities, embedding_dim=self.class_dim)
        self.modality_weight = nn.Parameter(torch.ones(flags.gnn_layers, self.num_modalities, 
                                                       self.num_modalities, self.num_heads)) # L, M, M, H

    def mod2id(self, name):
        return self.indices[name]
    
    def id2mod(self, id):
        for mod in self.indices.keys():
            if self.indices[mod] == id:
                return mod
        return None
    
    @property
    def device(self):
        if next(self.parameters(), None) is not None:
            return next(self.parameters()).device
        else:
            return torch.device('cpu')
    
    def get_visible_keys(self, exclude_modalities):
        state_dict_all = self.state_dict()
        state_dict_key = []
        for k in state_dict_all.keys():
            k_split = k.split('.')
            if k_split[0]=='encoders' or k_split[0]=='decoders' or k_split[0]=='decoders_single':
                if k_split[1] in exclude_modalities:
                    continue
            if k_split[0] == 'modality_embedding' or k_split[0] == 'modality_weight':
                continue
            state_dict_key.append(k)
        return state_dict_key

    def is_warmup_epoch(self, epoch):
        if epoch != -1 and epoch < self.flags.warmup_epoch:
            return True
        return False

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def set_fusion_functions(self):
        weights = utils.reweight_weights(torch.Tensor(self.flags.alpha_modalities))
        self.register_buffer('weight', weights)        
        # setting fusion methods before gnn
        if self.flags.fusion_method == 'zero':
            self.modality_fusion = self.zero_fusion
        elif self.flags.fusion_method == 'poe':
            self.modality_fusion = self.poe_fusion;
        elif self.flags.fusion_method == 'moe':
            self.modality_fusion = self.moe_fusion;
        self.calc_joint_divergence = self.divergence_static_prior

    def divergence_static_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights=self.weights;
        weights = weights.clone();
        weights = utils.reweight_weights(weights);
        div_measures = calc_group_divergence_moe(self.flags,
                                                 mus,
                                                 logvars,
                                                 weights,
                                                 normalization=mus.shape[1]);
        divs = dict();
        divs['joint_divergence'] = div_measures[0]; divs['individual_divs'] = div_measures[1]; divs['dyn_prior'] = None;
        return divs;

    def divergence_dynamic_prior(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights;
        div_measures = calc_alphaJSD_modalities(self.flags,
                                                mus,
                                                logvars,
                                                weights,
                                                normalization=self.flags.batch_size);
        divs = dict();
        divs['joint_divergence'] = div_measures[0];
        divs['individual_divs'] = div_measures[1];
        divs['dyn_prior'] = div_measures[2];
        return divs;

    def moe_fusion(self, mus, logvars, weights=None):
        if weights is None:
            weights = self.weights;
        weights = utils.reweight_weights(weights)
        mu_moe, logvar_moe = utils.mixture_component_selection(self.flags,
                                                               mus,
                                                               logvars,
                                                               weights);
        return [mu_moe, logvar_moe];

    def poe_fusion(self, mus, logvars, weights=None):
        mu_poe, logvar_poe = poe(mus, logvars);
        return [mu_poe, logvar_poe];

    def zero_fusion(self, mus, logvars, weights=None):
        return [torch.zeros_like(mus[0]), torch.zeros_like(logvars[0])]

    def clear_gnn_cache(self):
        for module in self.cm_graph_layers.modules():
            if isinstance(module, GCNConv):
                module._cached_edge_index = None
                module._cached_adj_t = None
    
    def gnn_forward(self, latents, mid):
        """
        'joint' with full modality, 'single' with batch_d modals
        cid and mid(int)
        """
        bs = latents['joint'][self.id2mod(mid)][0].shape[0]
        device = latents['joint'][self.id2mod(mid)][0].device

        node_feature = torch.Tensor().to(device) # bs, N, L
        node_modalities = []
        for m, m_key in enumerate(latents['joint'].keys()): # m0~m4
            if m == mid:
                node_feature = torch.cat((node_feature, 
                                            self.reparameterize(latents['joint'][m_key][0], 
                                                                latents['joint'][m_key][1])[:, None]), dim=1)
                impute_idx = node_feature.size(1) - 1
                node_modalities.append(m)
                continue

            if latents['single'].get(m_key, None) is not None:
                enc_feature = self.reparameterize(latents['single'][m_key][0], latents['single'][m_key][1])
                node_feature = torch.cat((node_feature, enc_feature[:, None]), dim=1)
                node_modalities.append(m)

        if self.flags.ablation_level == 3:
            # average of conditional modalities
            mask = torch.ones(node_feature.size(1), dtype=torch.bool)  # 全部初始化为 True
            mask[impute_idx] = False
            class_embeddings = node_feature[:, mask].mean(dim=1)
        elif self.flags.ablation_level == 4:
            # used with joint set to "poe" or "moe"
            class_embeddings = node_feature[:, impute_idx]
        else:
            node_modalities = torch.tensor(node_modalities).to(device)
            if self.flags.use_modality_embedding:
                node_feature += self.modality_embedding(node_modalities)[None] # 1, m, L
            edge_modal = construct_graph(node_feature.size(1), impute_idx, self.flags.ablation_level).to(device)
            for l, layer in enumerate(self.cm_graph_layers):
                if self.flags.use_modality_weight:
                    edge_weight = self.modality_weight[l, node_modalities[edge_modal[0]], node_modalities[edge_modal[1]]]
                    edge_weight[edge_weight < 0] = 0
                else:
                    edge_weight = None
                node_feature = layer(node_feature, edge_modal, edge_weight)
            class_embeddings = node_feature[:, impute_idx] # B, L
            mod_name = self.id2mod(mid)
            class_embeddings = class_embeddings.reshape(-1, self.mod_dimensions[mod_name], 
                                                            self.dim_multiples[mod_name]).mean(dim=2)
        return class_embeddings
    
    def forward(self, input_batch, subset_used, epoch=-1, *args, **kwargs):
        latents = self.inference(input_batch, subset_used);
        results = dict();
        results['latents'] = latents

        results['joint_divergence'] = 0 # no joint kl for gnn inputs
        
        div_single = self.calc_joint_divergence(latents['mus'],
                                                latents['logvars'],
                                                latents['weights_single'])
        for k, key in enumerate(div_single.keys()):
            results[key+'_single'] = div_single[key]

        results_rec = dict();
        results_rec_single = dict();
        enc_mods = latents['modalities'];
        for m, m_key in enumerate(input_batch.keys()):   
            # imputation   
            if not self.is_warmup_epoch(epoch):    
                m_s_mu, m_s_logvar = enc_mods[m_key + '_style'];
                if self.styles[m_key] > 0:
                    m_s_embeddings = self.reparameterize(mu=m_s_mu, logvar=m_s_logvar);
                else:
                    m_s_embeddings = None;
                
                class_embeddings = self.gnn_forward(latents, mid=self.mod2id(m_key))
                if hasattr(self.decoders[m_key], 'teacher_forcing'):
                    m_rec = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings, class_embeddings, input_batch[m_key]))
                else:
                    m_rec = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings, class_embeddings))
                results_rec[m_key] = m_rec

            # reconstruction
            if self.flags.k_single > 0:
                # class
                m_class_mu, m_class_logvar = latents['modalities'][m_key + '_single_dec']
                m_class_embeddings = self.reparameterize(mu=m_class_mu, logvar=m_class_logvar)
                # style
                m_style_mu, m_style_logvar = latents['modalities'][m_key + '_style' + '_single']
                m_s_embeddings_single = None
                if self.styles[m_key] > 0:
                    m_s_embeddings_single = self.reparameterize(mu=m_style_mu, logvar=m_style_logvar)
                # ! no additional decoders
                if hasattr(self.decoders[m_key], 'teacher_forcing'):
                    m_rec_single = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings_single, m_class_embeddings, input_batch[m_key]))
                else:
                    m_rec_single = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings_single, m_class_embeddings))
                results_rec_single[m_key] = m_rec_single
        results['rec'] = results_rec;
        results['rec_single'] = results_rec_single;
        return results;

    def encode(self, input_batch):
        latents = dict();
        for m, m_key in enumerate(self.modalities.keys()):
            if m_key in input_batch.keys():
                i_m = input_batch[m_key];
                l_single = self.encoders[m_key](i_m)
                l = l_single
                latents[m_key + '_style'] = l[:2]
                latents[m_key + '_style' + '_single'] = l_single[:2]
                latents[m_key] = [_.repeat(1, self.dim_multiples[m_key]).contiguous() for _ in l[2:4]] # Bs, CLS_DIM
                latents[m_key + '_single'] = [_.repeat(1, self.dim_multiples[m_key]).contiguous() for _ in l_single[2:4]]
                latents[m_key + '_single_dec'] = l_single[2:4]
            else:
                latents[m_key + '_style'] = [None, None]
                latents[m_key] = [None, None]
        return latents

    def inference(self, input_batch, subset_used, num_samples=None):
        latents = dict();
        enc_mods = self.encode(input_batch);
        latents['modalities'] = enc_mods

        # for single modalities
        mus = torch.Tensor().to(self.device);
        logvars = torch.Tensor().to(self.device);
        single = dict()
        for k, mod in enumerate(input_batch.keys()):
            mus = torch.cat((mus, enc_mods[mod][0].unsqueeze(0)), dim=0)
            logvars = torch.cat((logvars, enc_mods[mod][1].unsqueeze(0)), dim=0)
            single[mod] = [enc_mods[mod][0], enc_mods[mod][1]]
        
        # organize gnn input for the one to impute
        joint = dict()
        joint_mus = torch.Tensor().to(self.device);
        joint_logvars = torch.Tensor().to(self.device);
        for k, mod in enumerate(self.modalities.keys()):
            mus_subset = torch.Tensor().to(self.device);
            logvars_subset = torch.Tensor().to(self.device);
            for _k, _mod in enumerate(input_batch.keys()):
                if _mod == mod:
                    continue # no info leak
                mus_subset = torch.cat((mus_subset,enc_mods[_mod][0].unsqueeze(0)),dim=0)
                logvars_subset = torch.cat((logvars_subset, enc_mods[_mod][1].unsqueeze(0)), dim=0)
            if len(mus_subset) ==0:
                s_mu, s_logvar = [torch.zeros_like(enc_mods[_mod][0]), torch.zeros_like(enc_mods[_mod][1])]
            else:
                weights_subset = ((1/float(len(mus_subset)))*
                                        torch.ones(len(mus_subset)).to(self.device))
                s_mu, s_logvar = self.modality_fusion(mus_subset, logvars_subset, weights_subset) # moe, poe, zero
            
            joint[mod] = [s_mu, s_logvar]
            joint_mus = torch.cat((joint_mus, s_mu.unsqueeze(0)),dim=0)
            joint_logvars = torch.cat((joint_logvars, s_logvar.unsqueeze(0)), dim=0)

        weights = (1/float(joint_mus.shape[0]))*torch.ones(joint_mus.shape[0]).to(self.device);
        weights_single = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(self.device);
        latents['mus'] = mus; # M, B, L
        latents['logvars'] = logvars
        latents['single'] = single # ! only observed modality
        latents['weights_single'] = weights_single

        latents['joint'] = joint; # ! full modality
        latents['joint_mus'] = joint_mus; # used in kl comp
        latents['joint_logvars'] = joint_logvars; # used in kl comp
        latents['weights'] = weights

        latents['subsets_in_fusion'] = list(input_batch.keys())
        return latents;

    def generate(self, num_samples=None):
        # deprecated function?
        if num_samples is None:
            num_samples = self.flags.batch_size;
        z_styles = self.get_random_styles(num_samples);
        random_latents = {'content': dict(), 'style': z_styles};
        for mod in self.modalities.keys():
            mu = torch.zeros(num_samples,
                             self.mod_dimensions[mod]).to(self.device);
            logvar = torch.zeros(num_samples,
                                 self.mod_dimensions[mod]).to(self.device);
            z_class = self.reparameterize(mu, logvar);
            random_latents['content'][mod] = z_class
        random_samples = self.generate_from_latents(random_latents);
        return random_samples;

    def generate_sufficient_statistics_from_latents(self, latents, imputed_mods=None, target_sequence=None):
        suff_stats = dict();
        if imputed_mods is None:
            imputed_mods = self.modalities.keys()
        for m, m_key in enumerate(imputed_mods):
            c = latents['content'][m_key]
            s = latents['style'][m_key]
            if target_sequence is not None and hasattr(self.decoders[m_key], 'teacher_forcing'):
                cg = self.lhoods[m_key](*self.decoders[m_key](s, c, target_sequence[m_key]))
            else:
                cg = self.lhoods[m_key](*self.decoders[m_key](s, c))
            suff_stats[m_key] = cg;
        return suff_stats;

    def generate_from_latents(self, latents):
        suff_stats = self.generate_sufficient_statistics_from_latents(latents);
        cond_gen = dict();
        for m, m_key in enumerate(latents['style'].keys()):
            cond_gen_m = suff_stats[m_key].mean;
            cond_gen[m_key] = cond_gen_m;
        return cond_gen;

    def generate_from_distribution(self, distribution):
        latents = {'content': distribution['content'], 'style': {}}
        for k in distribution['style'].keys():
            if self.styles[k] > 0:
                latents['style'][k] = self.reparameterize(distribution['style'][k][0], distribution['style'][k][1])
            else:
                latents['style'][k] = None

        suff_stats = self.generate_sufficient_statistics_from_latents(latents);
        cond_gen = dict();
        for m, m_key in enumerate(latents['style'].keys()):
            cond_gen_m = suff_stats[m_key].mean
            cond_gen[m_key] = cond_gen_m;
        return cond_gen;

    def cond_generation(self, latents, num_samples=None):
        if num_samples is None:
            num_samples = latents['mus'][0].shape[0]

        style_latents = self.get_random_styles(num_samples);
        latents_for_decoder = {'content': dict(), 'style': style_latents}
        for m, m_key in enumerate(self.modalities.keys()): # m0-m4
            content_rep = self.gnn_forward(latents, m)
            latents_for_decoder['content'][m_key] = content_rep
        cond_gen_samples = self.generate_from_latents(latents_for_decoder)
        return cond_gen_samples

    def get_random_style_dists(self, num_samples):
        styles = dict();
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key];
            s_mu = torch.zeros(num_samples,
                               mod.style_dim).to(self.device)
            s_logvar = torch.zeros(num_samples,
                                   mod.style_dim).to(self.device);
            styles[m_key] = [s_mu, s_logvar];
        return styles;

    def get_random_styles(self, num_samples):
        styles = dict();
        for k, m_key in enumerate(self.modalities.keys()):
            if self.styles[m_key] > 0:
                mod = self.modalities[m_key];
                z_style = torch.randn(num_samples, mod.style_dim)
                z_style = z_style.to(self.device)
            else:
                z_style = None;
            styles[m_key] = z_style;
        return styles