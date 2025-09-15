import torch
import torch.multiprocessing as mp
from run_script.run_epochs_vaes import run_epochs_vae
from utils.main_comp import main_start, main_end
import argparse

if __name__ == '__main__':
    if torch.cuda.device_count() > 1:
        mp.set_start_method('spawn')
    dataset_parser = argparse.ArgumentParser()
    dataset_parser.add_argument('--dataset', type=str, default='', help="[polymnist, mnistsvhntext]")
    dataset_parser, remaining_argv = dataset_parser.parse_known_args()
    remaining_argv.append(f'--dataset={dataset_parser.dataset}')
    if dataset_parser.dataset == 'mnistsvhntext':
        from mnistsvhntext.flags import parser as mnistsvhntext_parser
        parser = mnistsvhntext_parser
    elif dataset_parser.dataset == 'polymnist':
        from mmnist.flags import parser as mnist_parser
        parser = mnist_parser
    elif dataset_parser.dataset == 'celeba':
        from celeba.flags import parser as celeba_parser
        parser = celeba_parser
    elif dataset_parser.dataset == 'eicu':
        from eicu.flags import parser as eicu_parser
        parser = eicu_parser
    elif dataset_parser.dataset == 'muse_eicu':
        from muse_eicu.flags import parser as eicu_parser
        from muse_eicu.eicu_dataset import eICUData
        parser = eicu_parser

    FLAGS = parser.parse_args(remaining_argv)
    assert FLAGS.dataset==dataset_parser.dataset
    FLAGS, mst = main_start(FLAGS)
    run_epochs_vae(mst)

    main_end(FLAGS)
