import os

import torch

from analysis import generate_prototype_activation_matrices
from datasets import get_datasets
from helpers import load_or_create_experiment
from model import construct_PPNet_for_config
from save import load_train_state

args, config, seed, DEBUG, load_state_path, checkpoint_file_prefix = load_or_create_experiment(force_load=True)

workers = 0 if DEBUG else 4

train_loader, train_push_loader, valid_loader, valid_push_loader, test_loader, test_push_loader = get_datasets(
    args.dataset, seed, workers, config)

print('Dataset loaded.')

ppnet = construct_PPNet_for_config(config).cuda()

(step,
 mode,
 epoch,
 iteration,
 experiment_run_name,
 best_accu,
 _) = load_train_state(load_state_path, ppnet, {})
print('Resuming experiment: {}'.format(experiment_run_name))
print('Best score so far: {}'.format(best_accu))
original_model_dir = os.path.dirname(load_state_path)

epoch = 50

fig_neg, fig_pos = generate_prototype_activation_matrices(ppnet, test_push_loader, train_push_loader, epoch,
                                                          original_model_dir, torch.device('cuda'))
fig_pos.savefig(os.path.join(original_model_dir, 'matrix_pos.png'))
fig_neg.savefig(os.path.join(original_model_dir, 'matrix_net.png'))
