import argparse
import getpass
import os
import random
import sys

import matplotlib
import numpy
import numpy as np
import torch

from save import get_state_path_for_prefix, load_config_from_train_state
from settings import Settings, get_settings

CHECKPOINT_PREFIX = 'checkpoint'
LOGS_DIR = 'runs'
SAVED_MODELS_PATH = 'saved_models'
CHECKPOINT_FREQUENCY_STEPS = 1


def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)


def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_or_create_experiment(force_load=False):
    matplotlib.use('Agg')

    # detect debugger
    gettrace = getattr(sys, 'gettrace', None)
    DEBUG = gettrace is not None and gettrace()
    if DEBUG:
        print('DEBUG MODE - parallelism disabled')

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(prog='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpuid', type=int, default=0, help='CUDA device id to use')
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=['mnist', 'colon_cancer', 'breast_cancer',
                                                                             'camelyon', 'messidor'], help='Select dataset')
    parser.add_argument('-n', '--new_experiment', default=False, action='store_true',
                        help='Overwrite any saved state and start a new experiment (saved checkpoint will be lost)')
    parser.add_argument('-l', '--load_state', metavar='STATE_FILE', type=str, default=None, required=force_load,
                        help='Continue training from specified state file (saved checkpoint will be lost)')
    parser.add_argument('-c', '--run_name_prefix', type=str, default=None, help='Prefix for the experiment name')
    parser.add_argument('-w', '--weighting_attention', default=False, action='store_true')
    parser.add_argument('--deterministic', type=str2bool, default=True, help='Use deterministic mode (slightly slower)')
    for param_name, param_type in Settings.as_params():
        parser.add_argument('--{}'.format(param_name), type=param_type)
    args = parser.parse_args()

    if args.new_experiment and args.load_state:
        print('You cannot load state and start a new experiment at the same time')
        exit(-1)

    checkpoint_file_prefix = '{}.{}.{}'.format(CHECKPOINT_PREFIX, args.dataset, getpass.getuser())
    checkpoint_files = get_state_path_for_prefix(checkpoint_file_prefix)
    load_state_path = None
    if args.load_state:
        load_state_path = args.load_state
    elif not args.new_experiment and len(checkpoint_files) > 0:
        if len(checkpoint_files) > 1:
            print('Multiple checkpoint files detected: {}'.format(checkpoint_files))
            print('Leave only one and remove all others to continue or specify which one to use with --load_state')
            exit(1)
        load_state_path = checkpoint_files[0]

    config = None
    if load_state_path:
        print('Loading state from: {}'.format(load_state_path))
        if args.deterministic:
            print('WARNING: resuming from saved state is not fully deterministic!')
        config = load_config_from_train_state(load_state_path)
    if config is None:
        print('Using default base settings for', args.dataset)
        config = get_settings(args.dataset)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    print('CUDA available:', torch.cuda.is_available())

    config = config.new_from_params(args)
    print(config)

    if config.random_seed_value is not None:
        seed = config.random_seed_value
    elif config.random_seed_id is not None and config.random_seed_id >= 0:
        seed = config.random_seed_presets[config.random_seed_id]
    else:
        seed = torch.seed()
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    print('Seed: {}'.format(seed))

    if args.deterministic:
        # Make deterministic.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.set_deterministic(True)
        print('Deterministic mode enabled.')
    else:
        print('WARNING: proceeding with non-deterministic mode.')

    return args, config, seed, DEBUG, load_state_path, checkpoint_file_prefix
