import argparse
import os

import torch

import prune
from datasets import get_datasets
from helpers import makedir, load_or_create_experiment
from model import construct_PPNet_for_config
from preprocess import preprocess_input_function
from save import load_train_state, save_train_state
from train_and_test import test, last_only, train, valid

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

optimize_last_layer = True

# pruning parameters
k = 6
prune_threshold = 6
find_threshold_prune_n_patches = 8
only_n_most_activated = None
# epoch = 50

original_model_dir = os.path.dirname(load_state_path)
original_model_name = os.path.basename(load_state_path)

assert not ('nopush' in original_model_name)

model_dir = os.path.join(original_model_dir, 'pruned_prototypes_epoch{}_k{}_pt{}'.format(epoch,
                                                                                         k,
                                                                                         prune_threshold))
makedir(model_dir)

# test(model=ppnet, dataloader=test_loader, config=config, step=0, weighting_attention=args.weighting_attention)

# prune prototypes
print('prune')
step += 1
prune.prune_prototypes(dataloader=train_push_loader,
                       ppnet=ppnet,
                       k=k,
                       prune_threshold=prune_threshold,
                       preprocess_input_function=preprocess_input_function,  # normalize
                       original_model_dir=original_model_dir,
                       epoch_number=epoch,
                       # model_name=None,
                       log=print,
                       copy_prototype_imgs=True,
                       find_threshold_prune_n_patches=find_threshold_prune_n_patches,
                       only_n_most_activated=only_n_most_activated)
accu = test(model=ppnet, dataloader=test_loader, config=config, weighting_attention=args.weighting_attention)
print('New best score: {}, saving snapshot'.format(accu))
save_train_state(os.path.join(model_dir, original_model_name.split('push')[0] + 'prune'), ppnet, {}, step, mode, epoch,
                 iteration, experiment_run_name, best_accu, accu, accu, config)
current_push_best_accu = accu

# last layer optimization
if optimize_last_layer:
    last_layer_optimizer_specs = [
        {
            'params': ppnet.last_layer.parameters(),
            'lr': config.last_layer_optimizer_lr['last_layer']
        },
        {
            'params': ppnet.attention_V.parameters(),
            'lr': config.last_layer_optimizer_lr['attention']
        },
        {
            'params': ppnet.attention_U.parameters(),
            'lr': config.last_layer_optimizer_lr['attention']
        },
        {
            'params': ppnet.attention_weights.parameters(),
            'lr': config.last_layer_optimizer_lr['attention']
        }
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    print('optimize last layer')

    for i in range(40):
        step += 1
        print('iteration: \t{0}'.format(i))
        last_only(model=ppnet)
        train(model=ppnet, dataloader=train_loader, optimizer=last_layer_optimizer, config=config,
              step=step, weighting_attention=args.weighting_attention)
        accu = valid(model=ppnet, dataloader=valid_loader, config=config, step=step,
                     weighting_attention=args.weighting_attention)
        if current_push_best_accu < accu:
            print('New best score: {}, saving snapshot'.format(accu))
            save_train_state(os.path.join(model_dir, original_model_name.split('push')[0] + 'prune'), ppnet, {}, step,
                             mode, epoch,
                             iteration, experiment_run_name, best_accu, accu, accu, config)
            current_push_best_accu = accu

test(model=ppnet, dataloader=test_loader, config=config, step=step,
     weighting_attention=args.weighting_attention)
