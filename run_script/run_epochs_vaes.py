import copy
import sys, os
import numpy as np
from itertools import cycle
import math
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import glog as logger
from collections import OrderedDict, defaultdict
import pickle
from divergence_measures.kl_div import calc_kl_divergence

from eval_metrics.coherence import test_generation_all, test_noise_imputation
from eval_metrics.representation import train_clf_lr_all_subsets, test_clf_lr_all_subsets
from eval_metrics.test_auprc import test_prc_all
from eval_metrics.test_mse import test_mse_all

from plotting import generate_plots

from utils import utils
from utils.TBLogger import TBLogger, dist_writer_process
from utils.BaseExperiment import BaseExperiment_impute
from utils.meter import Meter, MeterVector
from utils.dist_utils import synchronize_communication, setup, distribute_clients_to_cuda

# global variables
SEED = None 
SAMPLE1 = None
if SEED is not None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED) 

def calc_log_probs(exp, rec, batch_d):
    mods = exp.modalities
    log_probs = dict()
    weighted_log_prob = 0.0
    for m, m_key in enumerate(mods.keys()):
        if m_key in batch_d.keys():
            mod = mods[m_key]
            log_probs[mod.name] = -mod.calc_log_prob(rec[mod.name],
                                                    batch_d[mod.name],
                                                    batch_d[mod.name].shape[0])
            weighted_log_prob += exp.rec_weights[mod.name]*log_probs[mod.name]
    return log_probs, weighted_log_prob

def calc_klds(exp, result):
    if exp.flags.impute_method == 'graph':
        latents = result['latents']['single'];
    else:
        latents = result['latents']['subsets'];
    klds = dict();
    for m, key in enumerate(latents.keys()):
        mu, logvar = latents[key]
        klds[key] = calc_kl_divergence(mu, logvar,
                                       norm_value=mu.shape[0])
    return klds;

def calc_klds_style(exp, result, batch_d, single_prefix='_single'):
    mods = exp.modalities;
    latents = result['latents']['modalities'];
    klds = dict()
    weighted_klds = 0.0
    for m, m_key in enumerate(mods.keys()):
        if m_key in batch_d.keys():
            mu, logvar = latents[m_key+'_style'+single_prefix];
            if mu is not None:
                klds[m_key] = calc_kl_divergence(mu, logvar,
                                                          norm_value=mu.shape[0])
                weighted_klds += exp.style_weights[m_key] * klds[m_key]
    return klds, weighted_klds

def tmp_dict_func(f, d):
    out = {}
    for k in d.keys():
        if type(d[k][0])!=type(None):
            out[k] = [f(d[k][0]), f(d[k][1])]
        else:
            out[k] = [None, None]
    return out

def kl_annealing(epoch, kl_annealing):
    if kl_annealing == 0:
        return 1
    else:
        return min(epoch / kl_annealing, 1)

def basic_routine_epoch(epoch, exp:BaseExperiment_impute, batch, mm_vae, client_id, subset_used=None):
    # set up weights
    beta_style = exp.flags.beta_style
    beta_content = exp.flags.beta_content
    mods = exp.modalities
    
    # forward
    batch_all = batch[0]
    batch_l = batch[1]
    observed_mask = batch[2]
    batch_d = utils.mask_modalities(batch_all, observed_mask[0], exp.modalities_names)
    batch_d = {k: Variable(batch_d[k]).to(mm_vae.device) for k in batch_d.keys()}
    results = mm_vae(batch_d, subset_used, epoch)

    # STEP 1: loss of single modal vae
    if exp.flags.k_single > 0:
        log_probs_single_dict, weighted_log_prob_single = calc_log_probs(exp, results['rec_single'], batch_d)
        group_divergence_single = results['joint_divergence_single']
        weighted_kld_single = beta_content * group_divergence_single 
        # ! using same style kl in single and impute, only count in imputed
        total_loss_single = weighted_log_prob_single + weighted_kld_single * kl_annealing(epoch, exp.flags.kl_annealing)
    else:
        weighted_log_prob_single = weighted_kld_single = total_loss_single = 0
        log_probs_single_dict = dict()
    
    # STEP 2: loss of imputed vae
    if results['rec'] == dict():
        log_probs_dict, weighted_log_prob = dict(), 0
    else:
        log_probs_dict, weighted_log_prob = calc_log_probs(exp, results['rec'], batch_d);
    group_divergence = results['joint_divergence'] # KL_CONTENT = 0 when graph
    klds_dict = calc_klds(exp, results)
    klds_style_dict, weighted_kld_style = calc_klds_style(exp, results, batch_d, single_prefix='')
    if exp.flags.impute_method !='poe':
        kld_content = group_divergence;
        weighted_kld = beta_style * weighted_kld_style + beta_content * kld_content * kl_annealing(epoch, exp.flags.kl_annealing)
        total_loss_subsets = weighted_log_prob + weighted_kld
        total_loss = total_loss_subsets + exp.flags.k_single * total_loss_single
    else: # ! poe always use unimodal elbos
        klds_joint = {'content': group_divergence * kl_annealing(epoch, exp.flags.kl_annealing),
                      'style': dict()}
        elbos = dict()
        weighted_kld = 0
        weighted_log_prob = 0
        for m, m_key in enumerate(batch_d.keys()):
            mod = mods[m_key]
            kld_style_m = klds_style_dict.get(m_key, 0)
            klds_joint['style'][m_key] = kld_style_m
            i_batch_mod = {m_key: batch_d[m_key]}
            r_mod = mm_vae(i_batch_mod, subset_used)
            log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                              batch_d[m_key],
                                              batch_d[m_key].shape[0]);
            log_prob = {m_key: log_prob_mod}
            klds_mod = {'content': klds_dict[m_key] * kl_annealing(epoch, exp.flags.kl_annealing)}
            elbo_mod, div, rec_error = utils.calc_elbo(exp, m_key, log_prob, klds_mod);
            elbos[m_key] = elbo_mod
            weighted_kld += div.item()/(len(batch_d.keys())+1)
            weighted_log_prob += rec_error.item()/(len(batch_d.keys())+1)

        elbo_joint, div, rec_error = utils.calc_elbo(exp, 'joint', log_probs_dict, klds_joint);
        elbos['joint'] = elbo_joint;
        total_loss = sum(elbos.values())
        weighted_kld += div.item()/(len(batch_d.keys())+1)
        weighted_log_prob += rec_error.item()/(len(batch_d.keys())+1)
        total_loss_subsets = 0

    out_basic_routine = dict();
    out_basic_routine['results'] = results;
    out_basic_routine['log_probs'] = log_probs_dict;
    out_basic_routine['total_loss'] = total_loss;
    out_basic_routine['total_loss_subsets'] = total_loss_subsets;
    out_basic_routine['klds_content'] = klds_dict;
    out_basic_routine['klds_style'] = klds_style_dict;
    out_basic_routine['kld_weighted'] = weighted_kld;
    out_basic_routine['weighted_log_prob'] = weighted_log_prob;

    if exp.flags.k_single > 0:
        out_basic_routine['weighted_log_prob_single'] = weighted_log_prob_single
        out_basic_routine['log_probs_single'] = log_probs_single_dict
        out_basic_routine['kld_weighted_single'] = weighted_kld_single
        out_basic_routine['total_loss_single'] = exp.flags.k_single*total_loss_single
    return out_basic_routine;

def print_log(name, iteration, total_iter, loss, klds, log_probs, joint_divergence):
    str_out = '%s Iteration %d/%d; loss: %.4f; joint_divergence: %.4f; '%(name, iteration, total_iter, loss, joint_divergence)    
    str_out += 'kld:'
    for k in klds.keys():
        str_out += ' %s(%.4f)'%(k, klds[k].item())    
    str_out += '; log probs:'
    for k in log_probs.keys():
        str_out += ' %s(%.4f)'%(k, log_probs[k].item())    
    logger.info(str_out)

def dict2str(out_dict):
    str_out = ''
    for i, k in enumerate(out_dict.keys()):
        str_out += str(k)+':'
        if isinstance(out_dict[k], dict):
            for k_k in out_dict[k].keys():
                str_out += ' %.6f (%s)'%(out_dict[k][k_k], k_k)
        else:
            str_out += ' %.6f'%(out_dict[k])
        
        str_out += '; '
    return str_out

def cosine_annealing_lr(initial_lr, epoch, max_epochs):
    min_lr = initial_lr / 100
    lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
    return lr

def local_round_summarization(rank, logger_dict_dict, exp, epoch, queue, mode='Train'):
    for k, v in logger_dict_dict.items():
        queue.put({k: v})
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        logger_dict_list = []
        data_length = []
        while not queue.empty():
            logger_dict = queue.get()
            c_i = list(logger_dict.keys())[0]
            logger_dict = list(logger_dict.values())[0]
            data_length.append(len(exp.train_impute_dataset[c_i]))
            logger_dict_list.append(logger_dict)
        if len(logger_dict_list) != 0:
            avg_log_dict = utils.avg_dict(logger_dict_list, data_length)
            logger.info(f'-------{mode} epoch {epoch} all: ' + dict2str(avg_log_dict)) 
        else:
            logger.warn(f'-------{mode} epoch {epoch} all: Failed to gather infos!')         
    torch.cuda.synchronize()
    dist.barrier()

@torch.no_grad()
def gather_local_state(exp, models, clients_sample):
    update_state = {k: torch.zeros_like(v) for k, v in models[clients_sample[0]].state_dict().items()}
    state_count = [0 for _ in models[clients_sample[0]].state_dict()]
    for c_i in clients_sample:
        local_state = models[c_i].state_dict()
        invisible_modalities = utils.get_invisible_modalities(exp.observed_mask[c_i], exp.modalities_names)
        visible_keys = models[c_i].get_visible_keys(invisible_modalities)
        for i, key in enumerate(local_state):
            if key not in visible_keys:
                continue
            update_state[key] += local_state[key]
            state_count[i] += 1
    return update_state, torch.tensor(state_count, device=models[clients_sample[0]].device)

@torch.no_grad()
def gather_modality_state(exp, models, clients_sample):
    modality_embedding = torch.zeros_like(models[clients_sample[0]].modality_embedding.weight)
    modality_weight = torch.zeros_like(models[clients_sample[0]].modality_weight)
    for c_i in clients_sample:
        observed_mask = exp.observed_mask[c_i]
        available_mod = torch.nonzero(observed_mask).squeeze()
        modality_embedding[observed_mask] += models[c_i].modality_embedding.weight[observed_mask]
        modality_weight[:, available_mod[:, None], available_mod] += models[c_i].modality_weight[:, available_mod[:, None], available_mod]
    return modality_embedding, modality_weight

@synchronize_communication
def global_communication(update_state, *args):
    for key in update_state.keys():
        dist.all_reduce(update_state[key], op=dist.ReduceOp.SUM)
    for arg in args:
        dist.all_reduce(arg, op=dist.ReduceOp.SUM)

@torch.no_grad()
def global_round_communication(rank, models, clients_sample, exp):
    update_state, state_count = gather_local_state(exp, models, clients_sample)
    if exp.flags.impute_method == 'graph':
        modality_embedding, modality_weight = gather_modality_state(exp, models, clients_sample)
    modality_counts = exp.observed_mask.sum(dim=0).to(models[clients_sample[0]].device)
    modality_cooccur = exp.observed_mask.to(torch.float).T @ exp.observed_mask.to(torch.float) # M, M
    modality_cooccur = modality_cooccur.to(models[clients_sample[0]].device)
    if rank != -1: # multi-device
        if exp.flags.impute_method == 'graph':
            global_communication(update_state, state_count, modality_embedding, modality_weight)
        else:
            global_communication(update_state, state_count)
    for i, key in enumerate(models[clients_sample[0]].state_dict()):
        if key in update_state:
            update_state[key] = update_state[key] / state_count[i]
    if exp.flags.impute_method == 'graph':
        update_state['modality_embedding.weight'] = modality_embedding / (modality_counts[:, None] + 1e-8)
        update_state['modality_weight'] = modality_weight / (modality_cooccur[None, :, :, None] + 1e-8)
    return update_state

def device_train_routine(rank, world_size, epoch, gpu2client, exp, queue: mp.Queue):
    # ! logger(distributed?) & pickle record
    models = OrderedDict()
    optims = OrderedDict()
    lr = cosine_annealing_lr(exp.flags.initial_learning_rate, epoch, exp.flags.end_epoch)
    for c_i in gpu2client[rank]:
        if len(gpu2client[rank]) > 1:
            models[c_i] = copy.deepcopy(exp.model)
            for module in models[c_i].modules():
                if isinstance(module, nn.modules.RNNBase):
                    module.flatten_parameters()
        else:
            models[c_i] = exp.model
        models[c_i].train()
        optims[c_i] = exp.optimizer_class(models[c_i].parameters(), lr=lr, weight_decay=1e-6)  
    for e in range(epoch, min(epoch + exp.flags.communication_freq, exp.flags.end_epoch)):
        if rank == 0:
            logger.info('------Epoch %d/%d---------'%(e, exp.flags.end_epoch))
        client_log_dict_dict = dict()
        for c_i in gpu2client[rank]:
            subset_used = exp.clients_subsets[c_i]
            model = models[c_i]            
            optimizer = optims[c_i]
            meter_client = Meter()
            for iteration, batch in enumerate(exp.train_dataloader[c_i]):
                optimizer.zero_grad()                
                basic_routine = basic_routine_epoch(e, exp, batch, model, c_i, subset_used);
                total_loss = basic_routine['total_loss']
                log_probs = basic_routine['log_probs']
                klds_dict = basic_routine['klds_content']
                klds_style_dict = basic_routine['klds_style']
                # backprop                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()

                meter_client._update({'loss': total_loss, 
                                      'kld': basic_routine['kld_weighted'], 
                                      'log_prob': basic_routine['weighted_log_prob'],
                                      'loss_subsets': basic_routine['total_loss_subsets']},
                                      batch_size=batch[2].shape[0])
                meter_client._update({'loss_prob_'+m: v for m, v in log_probs.items()},
                                     batch_size=batch[2].shape[0])
                meter_client._update({'klds_content_'+m: v for m, v in klds_dict.items()},
                                     batch_size=batch[2].shape[0])
                meter_client._update({'klds_style_'+m: v for m, v in klds_style_dict.items()},
                                     batch_size=batch[2].shape[0])
                if exp.flags.k_single > 0:
                    log_probs_single = basic_routine['log_probs_single']
                    meter_client._update({
                                      'loss_sin': basic_routine['total_loss_single'], 
                                      'kld_sin': basic_routine['kld_weighted_single'], 
                                      'log_prob_sin': basic_routine['weighted_log_prob_single']},
                                      batch_size=batch[2].shape[0])
                    meter_client._update({'loss_prob_single_'+m: v for m, v in log_probs_single.items()},
                                        batch_size=batch[2].shape[0])
            logger.info('Train epoch %d, client %d on device %d: '%(e, c_i, rank) + str(meter_client))
            client_log_dict_dict[c_i] = meter_client.get_scalar_dict('global_avg')
        
        # gather from local clients
        local_round_summarization(rank, client_log_dict_dict, exp, e, queue)
        torch.cuda.empty_cache()

    updated_dict = global_round_communication(rank, models, gpu2client[rank], exp)
    exp.model.load_state_dict(updated_dict, strict=True)
    torch.cuda.empty_cache()

@torch.no_grad()
def device_test_routine(rank, world_size, epoch, gpu2client, exp, queue):
    mm_vae = exp.model
    mm_vae.eval()
    exp.model = mm_vae
    for c_i in gpu2client[rank]:
        subset_used = exp.clients_subsets[c_i]
        meter_client = Meter()
        client_log_dict_dict = dict()
        for iteration, batch in enumerate(exp.test_dataloader[c_i]):
            basic_routine = basic_routine_epoch(epoch, exp, batch, mm_vae, c_i, subset_used);
            total_loss = basic_routine['total_loss'];
            log_probs = basic_routine['log_probs'];
            klds_dict = basic_routine['klds_content']
            klds_style_dict = basic_routine['klds_style']
            meter_client._update({'loss': total_loss, 
                                'kld': basic_routine['kld_weighted'], 
                                'log_prob': basic_routine['weighted_log_prob'],
                                'loss_subsets': basic_routine['total_loss_subsets']},
                                batch_size=batch[2].shape[0])
            meter_client._update({'loss_prob_'+m: v for m, v in log_probs.items()},
                                    batch_size=batch[2].shape[0])
            meter_client._update({'klds_content_'+m: v for m, v in klds_dict.items()},
                                     batch_size=batch[2].shape[0])
            meter_client._update({'klds_style_'+m: v for m, v in klds_style_dict.items()},
                                     batch_size=batch[2].shape[0])
            if exp.flags.k_single > 0:
                log_probs_single = basic_routine['log_probs_single']
                meter_client._update({
                                    'loss_sin': basic_routine['total_loss_single'], 
                                    'kld_sin': basic_routine['kld_weighted_single'], 
                                    'log_prob_sin': basic_routine['weighted_log_prob_single']},
                                    batch_size=batch[2].shape[0])
                meter_client._update({'loss_prob_single_'+m: v for m, v in log_probs_single.items()},
                                    batch_size=batch[2].shape[0])
        logger.info('Test epoch %d, client %d on device %d: '%(epoch, c_i, rank) + str(meter_client))
        client_log_dict_dict[c_i] = meter_client.get_scalar_dict('global_avg')
    local_round_summarization(rank, client_log_dict_dict, exp, epoch, queue, mode='Test')
    torch.cuda.empty_cache()

def run_epochs_mp(rank, world_size, gpu2client, exp: BaseExperiment_impute, queue: mp.Queue):
    setup(exp, rank, world_size)
    if rank == 0:
        writer = SummaryWriter(exp.flags.dir_logs)
        tb_logger = TBLogger(os.path.basename(exp.flags.dir_experiment), writer)
        str_flags = utils.save_and_log_flags(exp.flags)
        tb_logger.writer.add_text('FLAGS', str_flags, 0)
    lastest_checkpoint = exp.flags.start_epoch
    last_val = exp.flags.start_epoch
    exp.to(rank)
    exp.init_dataloader(gpu2client[rank])
    for epoch in range(exp.flags.start_epoch, exp.flags.end_epoch, exp.flags.communication_freq):
        device_train_routine(rank, world_size, epoch, gpu2client, exp, queue)
        
        if (epoch+exp.flags.communication_freq-last_val) >= exp.flags.validation_freq \
                or (epoch + exp.flags.communication_freq) >= exp.flags.end_epoch:
            device_test_routine(rank, world_size, 
                                min(epoch+exp.flags.communication_freq, exp.flags.end_epoch)-1, 
                                gpu2client, exp, queue)
            last_val = epoch+exp.flags.communication_freq
        
        if rank == 0:
            if (epoch+exp.flags.communication_freq-lastest_checkpoint) >= exp.flags.eval_freq \
                or (epoch + exp.flags.communication_freq) >= exp.flags.end_epoch:
                # save ckpt
                torch.save(exp.model.state_dict(),
                            os.path.join(exp.flags.dir_checkpoints, str(epoch + exp.flags.communication_freq-1).zfill(4)))
                checkpoints = sorted([f for f in os.listdir(exp.flags.dir_checkpoints) if f.isdigit()])
                if len(checkpoints) > 5:
                    for ckpt in checkpoints[:-5]:
                        ckpt_path = os.path.join(exp.flags.dir_checkpoints, ckpt)
                        os.remove(ckpt_path)

                # writer process
                try:
                    dist_writer_process(tb_logger, exp, start_epoch=lastest_checkpoint)
                except:
                    pass
                
                # evaluation
                test_metric(epoch+exp.flags.communication_freq-1, exp, tb_logger)
                lastest_checkpoint = epoch+exp.flags.communication_freq
        torch.cuda.synchronize()
        dist.barrier()
    if rank == 0:
        tb_logger.writer.close()

def dist_train(exp, world_size):
    clients_sample = list(range(exp.flags.client_num))
    gpu2client = distribute_clients_to_cuda(clients_sample, world_size)
    queue = mp.Queue()
    mp.spawn(run_epochs_mp, 
             args=(world_size, gpu2client, exp, queue), 
             nprocs=world_size, join=True)
    queue.close()
    queue.join_thread()

def train(epoch, exp: BaseExperiment_impute, tb_logger: TBLogger):
    # ! full clients
    clients_sample = list(range(exp.flags.client_num))
    models = OrderedDict()
    optims = OrderedDict()
    lr = cosine_annealing_lr(exp.flags.initial_learning_rate, epoch, exp.flags.end_epoch)
    for c_i in clients_sample:
        if c_i >= 1:
            models[c_i] = copy.deepcopy(exp.model).to(exp.flags.device)
            for module in models[c_i].modules():
                if isinstance(module, nn.modules.RNNBase):
                    module.flatten_parameters()
        else:
            exp.model = exp.model.to(exp.flags.device)
            models[c_i] = exp.model
        models[c_i].train()
        optims[c_i] = exp.optimizer_class(models[c_i].parameters(), lr=lr, weight_decay=1e-6)
    client_log_dict_list = dict()
    for e in range(epoch, epoch + exp.flags.communication_freq):
        exp.pickle_record['train'][e] = {}
        client_log_dict_list[e] = []
    for c_i in clients_sample:
        for e in range(epoch, epoch + exp.flags.communication_freq):
            subset_used = exp.clients_subsets[c_i]
            model = models[c_i]            
            optimizer = optims[c_i]
            meter_client = Meter()
            for iteration, batch in enumerate(exp.train_dataloader[c_i]):
                if hasattr(exp, 'trainset_samples') and exp.trainset_samples <= exp.flags.batch_size*iteration:
                    break
                optimizer.zero_grad()
                basic_routine = basic_routine_epoch(e, exp, batch, model, c_i, subset_used)
                total_loss = basic_routine['total_loss']
                log_probs = basic_routine['log_probs']
                klds_dict = basic_routine['klds_content']
                klds_style_dict = basic_routine['klds_style']
                # backprop
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                meter_client._update({'loss': total_loss, 
                                      'kld': basic_routine['kld_weighted'], 
                                      'log_prob': basic_routine['weighted_log_prob'],
                                      'loss_subsets': basic_routine['total_loss_subsets']},
                                      batch_size=batch[2].shape[0])
                meter_client._update({'loss_prob_'+m: v for m, v in log_probs.items()},
                                     batch_size=batch[2].shape[0])
                meter_client._update({'klds_content_'+m: v for m, v in klds_dict.items()},
                                     batch_size=batch[2].shape[0])
                meter_client._update({'klds_style_'+m: v for m, v in klds_style_dict.items()},
                                     batch_size=batch[2].shape[0])
                if exp.flags.k_single > 0:
                    log_probs_single = basic_routine['log_probs_single']
                    meter_client._update({
                                      'loss_sin': basic_routine['total_loss_single'], 
                                      'kld_sin': basic_routine['kld_weighted_single'], 
                                      'log_prob_sin': basic_routine['weighted_log_prob_single']},
                                      batch_size=batch[2].shape[0])
                    meter_client._update({'loss_prob_single_'+m: v for m, v in log_probs_single.items()},
                                        batch_size=batch[2].shape[0])
            # logging info
            logger.info('Train epoch %d, client %d: '%(e, c_i) + str(meter_client))
            logger_dict = meter_client.get_scalar_dict('global_avg')
            client_log_dict_list[e].append(logger_dict)
            exp.pickle_record['train'][e][c_i] = logger_dict
        del optims[c_i]
        models[c_i] = models[c_i].to('cpu')
        torch.cuda.empty_cache()

    for e in range(epoch, epoch + exp.flags.communication_freq):
        avg_log_dict = utils.avg_dict(client_log_dict_list[e], exp.client_train_dataset_len)
        logger.info('-------Train epoch %d all: '%(e) + dict2str(avg_log_dict))
        tb_logger.write_dict({'train/'+k+'_all':avg_log_dict[k] for k in avg_log_dict.keys()}, step=e)

    update_state = global_round_communication(-1, models, clients_sample, exp)
    exp.model.load_state_dict(update_state, strict=False)

@torch.no_grad()
def test(epoch, exp, tb_logger):
    mm_vae = exp.model.to(exp.flags.device);
    mm_vae.eval();
    exp.model = mm_vae;

    exp.pickle_record['test'][epoch] = {}
    exp.pickle_record['test_metric'][epoch] = {}

    client_log_dict_list = []
    for c_i in range(exp.flags.client_num):
        subset_used = exp.clients_subsets[c_i]

        meter_client = Meter()
        for iteration, batch in enumerate(exp.test_dataloader[c_i]):
            basic_routine = basic_routine_epoch(epoch, exp, batch, mm_vae, c_i, subset_used);
            results = basic_routine['results']
            total_loss = basic_routine['total_loss']
            log_probs = basic_routine['log_probs']
            klds_dict = basic_routine['klds_content']
            klds_style_dict = basic_routine['klds_style']

            meter_client._update({'loss': total_loss, 
                                      'kld': basic_routine['kld_weighted'], 
                                      'log_prob': basic_routine['weighted_log_prob'],
                                      'loss_subsets': basic_routine['total_loss_subsets']},
                                      batch_size=batch[2].shape[0])
            meter_client._update({'loss_prob_'+m: v for m, v in log_probs.items()},
                                    batch_size=batch[2].shape[0])
            meter_client._update({'klds_content_'+m: v for m, v in klds_dict.items()},
                                     batch_size=batch[2].shape[0])
            meter_client._update({'klds_style_'+m: v for m, v in klds_style_dict.items()},
                                     batch_size=batch[2].shape[0])
            if exp.flags.k_single > 0:
                log_probs_single = basic_routine['log_probs_single']
                meter_client._update({
                                    'loss_sin': basic_routine['total_loss_single'], 
                                    'kld_sin': basic_routine['kld_weighted_single'], 
                                    'log_prob_sin': basic_routine['weighted_log_prob_single']},
                                    batch_size=batch[2].shape[0])
                meter_client._update({'loss_prob_single_'+m: v for m, v in log_probs_single.items()},
                                    batch_size=batch[2].shape[0])

        logger.info('Test epoch %d, client %d: '%(epoch, c_i) + str(meter_client))
        logger_dict = meter_client.get_scalar_dict('global_avg')
        # tb_logger.write_dict({'test/'+k+'_client%d'%c_i:logger_dict[k] for k in logger_dict.keys()}, step=epoch)
        # tb_logger.write_latent_distr('test', results['latents']);
        client_log_dict_list.append(logger_dict)

        exp.pickle_record['test'][epoch][c_i] = logger_dict

    avg_log_dict = utils.avg_dict(client_log_dict_list, exp.client_test_dataset_len)
    logger.info('-------Test epoch %d all: '%(epoch) + dict2str(avg_log_dict))
    tb_logger.write_dict({'test/'+k+'_all':avg_log_dict[k] for k in avg_log_dict.keys()}, step=epoch)
    tb_logger.write_latent_distr('test', results['latents'], step=epoch);

@torch.no_grad()
def test_metric(epoch, exp, tb_logger=None):
    exp.model = exp.model.eval().to(exp.flags.device)
    if exp.flags.dataset!='eicu' and exp.flags.dataset!='muse_eicu':
        plots = generate_plots(exp, epoch);
        if tb_logger:
            tb_logger.write_plots(plots, epoch);
        
    if epoch not in exp.pickle_record['test_metric'].keys():
        exp.pickle_record['test_metric'][epoch] = {}

    if exp.flags.noisy_impute and exp.flags.use_clf:
        test_noise_imputation(epoch, exp)
        return

    if exp.flags.calc_mse or (exp.flags.dataset=='eicu' and epoch == exp.flags.end_epoch-1):
        mse_eval = test_mse_all(epoch, exp)
        if tb_logger:
            tb_logger.write_scalars('MSE', mse_eval, step=epoch)
        logger.info('-------Test epoch %d: '%(epoch) + dict2str({'MSE': mse_eval}))

    if exp.flags.eval_lr or epoch == exp.flags.end_epoch-1:
        clf_lr = train_clf_lr_all_subsets(exp);
        lr_eval = test_clf_lr_all_subsets(epoch, clf_lr, exp)
        if tb_logger:
            tb_logger.write_lr_eval(lr_eval, step=epoch);
        logger.info('-------Test epoch %d: '%(epoch) + dict2str({'latent classification': lr_eval}))
        exp.pickle_record['test_metric'][epoch]['latent_classification'] = lr_eval

    if exp.flags.use_clf or (exp.flags.dataset!='eicu' and exp.flags.dataset!='muse_eicu' and epoch == exp.flags.end_epoch-1):
        with torch.no_grad():
            gen_eval = test_generation_all(epoch, exp);
        if tb_logger:
            tb_logger.write_coherence_logs_all(gen_eval, step=epoch);
        logger.info('-------Test epoch %d, Coherence: '%(epoch) + ' cond | ' + dict2str(gen_eval['cond']))
        exp.pickle_record['test_metric'][epoch]['coherence'] = gen_eval

    if exp.flags.calc_auprc or (exp.flags.dataset=='muse_eicu' and epoch == exp.flags.end_epoch-1):
        with torch.no_grad():
            auprc = test_prc_all(epoch, exp)
        if tb_logger:
            tb_logger.write_scalars('AUPRC', auprc, step=epoch)
        logger.info('-------Test epoch %d: '%(epoch) + dict2str({'AUPRC': auprc}))
        exp.pickle_record['test_metric'][epoch]['auprc'] = auprc

def run_epochs_vae(exp: BaseExperiment_impute):
    # test mode
    if exp.flags.test_only:
        writer = SummaryWriter(exp.flags.dir_logs)
        tb_logger = TBLogger(os.path.basename(exp.flags.dir_experiment), writer)
        str_flags = utils.save_and_log_flags(exp.flags)
        tb_logger.writer.add_text('FLAGS', str_flags, 0)
        exp.to(exp.flags.device)
        test_metric(-1, exp, tb_logger);
        return

    world_size = min(exp.flags.client_num, torch.cuda.device_count())
    logger.info(f'training epochs progress with {world_size} GPUS: ')
    
    # distributed training with lifelong process
    if world_size > 1:
        dist_train(exp, world_size)
        return

    # single process training
    writer = SummaryWriter(exp.flags.dir_logs)
    tb_logger = TBLogger(os.path.basename(exp.flags.dir_experiment), writer)
    str_flags = utils.save_and_log_flags(exp.flags)
    tb_logger.writer.add_text('FLAGS', str_flags, 0)
    exp.to(exp.flags.device)
    exp.init_dataloader(list(range(exp.flags.client_num)))
    lastest_checkpoint = exp.flags.start_epoch - 1
    last_val = exp.flags.start_epoch - 1
    for epoch in range(exp.flags.start_epoch, exp.flags.end_epoch, exp.flags.communication_freq):
        train(epoch, exp, tb_logger)
        if (epoch+exp.flags.communication_freq-last_val) > exp.flags.validation_freq \
                or (epoch + exp.flags.communication_freq) >= exp.flags.end_epoch:
            last_val = epoch+exp.flags.communication_freq-1
            test(min(epoch+exp.flags.communication_freq, exp.flags.end_epoch)-1, exp, tb_logger)

        with open(os.path.join(exp.flags.target_dir, 'pickle.pkl'), "wb") as f:
            pickle.dump(exp.pickle_record, f)

        if (epoch+exp.flags.communication_freq-lastest_checkpoint) > exp.flags.eval_freq \
            or (epoch + exp.flags.communication_freq) >= exp.flags.end_epoch:
            torch.save(exp.model.state_dict(),
                       os.path.join(exp.flags.dir_checkpoints, str(epoch + exp.flags.communication_freq-1).zfill(4)))
            checkpoints = sorted([f for f in os.listdir(exp.flags.dir_checkpoints) if f.isdigit()])
            if len(checkpoints) > 5:
                for ckpt in checkpoints[:-5]:
                    ckpt_path = os.path.join(exp.flags.dir_checkpoints, ckpt)
                    os.remove(ckpt_path)
            test_metric(epoch+exp.flags.communication_freq-1, exp, tb_logger)
            lastest_checkpoint = epoch+exp.flags.communication_freq-1