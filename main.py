import argparse
import datetime
import os
import platform

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from datasets.colon_dataset import ColonCancerBagsCross
from datasets.mnist_dataset import MnistBags
from helpers import makedir
from model import construct_PPNet
from preprocess import preprocess_input_function
from push import push_prototypes
from save import load_train_state, save_train_state, get_state_path_for_prefix, snapshot_code
from settings import COLON_CANCER_SETTINGS, MNIST_SETTINGS
from train_and_test import warm_only, train, joint, test, last_only, TrainMode

CHECKPOINT_PREFIX = 'checkpoint'
LOGS_DIR = 'runs'
SAVED_MODELS_PATH = 'saved_models'
CHECKPOINT_FREQUENCY_STEPS = 3

# noinspection PyTypeChecker
parser = argparse.ArgumentParser(prog='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-g', '--gpuid', type=int, default=0, help='CUDA device id to use')
parser.add_argument('-d', '--dataset', type=str, default='colon_cancer', choices=['mnist', 'colon_cancer'],
                    help='Select dataset')
parser.add_argument('-n', '--new_experiment', default=False, action='store_true',
                    help='Overwrite any saved state and start a new experiment (saved checkpoint will be lost)')
parser.add_argument('-l', '--load_state', metavar='STATE_FILE', type=str, default=None,
                    help='Continue training from specified state file (saved checkpoint will be lost)')
args = parser.parse_args()

if args.new_experiment and args.load_state:
    print('You cannot load state and start a new experiment at the same time')
    exit(-1)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
print('CUDA available:', torch.cuda.is_available())

if args.dataset == 'colon_cancer':
    split_val = 70
    train_range, test_range = range(split_val), range(split_val, 100)

    ds = ColonCancerBagsCross(path="data/ColonCancer", train=True, train_val_idxs=train_range, test_idxs=test_range,
                              shuffle_bag=True)
    ds_push = ColonCancerBagsCross(path="data/ColonCancer", train=True, train_val_idxs=train_range,
                                   test_idxs=test_range,
                                   push=True, shuffle_bag=True)
    ds_test = ColonCancerBagsCross(path="data/ColonCancer", train=False, train_val_idxs=train_range,
                                   test_idxs=test_range)
    config = COLON_CANCER_SETTINGS
elif args.dataset == 'mnist':
    ds = MnistBags(train=True)
    ds_push = MnistBags(train=True)
    ds_test = MnistBags(train=False)
    config = MNIST_SETTINGS
else:
    raise NotImplementedError()

ppnet = construct_PPNet(base_architecture=config.base_architecture,
                        pretrained=False, img_size=config.img_size,
                        prototype_shape=config.prototype_shape,
                        num_classes=config.num_classes,
                        prototype_activation_function=config.prototype_activation_function,
                        add_on_layers_type=config.add_on_layers_type)
ppnet = ppnet.cuda()

summary(ppnet, (1, 3, config.img_size, config.img_size))

joint_optimizer_specs = [
    {
        'params': ppnet.features.parameters(),
        'lr': config.joint_optimizer_lrs['features'],
        'weight_decay': 1e-3
    },
    {
        'params': ppnet.add_on_layers.parameters(),
        'lr': config.joint_optimizer_lrs['add_on_layers'],
        'weight_decay': 1e-3
    },
    {
        'params': ppnet.prototype_vectors,
        'lr': config.joint_optimizer_lrs['prototype_vectors']
    }
]

warm_optimizer_specs = [
    {
        'params': ppnet.add_on_layers.parameters(),
        'lr': config.warm_optimizer_lrs['add_on_layers'],
        'weight_decay': 1e-3
    },
    {
        'params': ppnet.prototype_vectors,
        'lr': config.warm_optimizer_lrs['prototype_vectors']
    },
]

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

joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=config.joint_lr_step_size, gamma=0.1)
warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

other_state = {
    'joint_optimizer': joint_optimizer,
    'joint_lr_scheduler': joint_lr_scheduler,
    'warm_optimizer': warm_optimizer,
    'last_layer_optimizer': last_layer_optimizer,
}

checkpoint_file_prefix = '{}.{}'.format(CHECKPOINT_PREFIX, args.dataset)
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
if load_state_path:
    print('Loading state from: {}'.format(load_state_path))
    (step,
     mode,
     epoch,
     iteration,
     experiment_run_name,
     best_accu,
     current_push_best_accu) = load_train_state(load_state_path, ppnet, other_state)
    print('Resuming experiment: {}'.format(experiment_run_name))
    print('Best score so far: {}'.format(best_accu))
else:
    step = 0
    mode = TrainMode.WARM
    epoch = 0
    iteration = None
    experiment_run_name = '{}.{}.{}'.format(args.dataset, platform.node(), datetime.datetime.now().isoformat())
    best_accu = 0.
    current_push_best_accu = None
    print('Starting new experiment: {}'.format(experiment_run_name))
    print('Saving code snapshot with git-experiments')
    snapshot_code(experiment_run_name)

model_dir = os.path.join(SAVED_MODELS_PATH, experiment_run_name)
makedir(model_dir)
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)

log_writer = SummaryWriter(os.path.join(LOGS_DIR, experiment_run_name), purge_step=step + 1)

weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

seed = torch.seed()
torch.manual_seed(seed)

log_writer.add_text('seed', str(seed))

train_loader = torch.utils.data.DataLoader(
    ds, batch_size=None, shuffle=True,
    num_workers=8,
    pin_memory=True)
train_push_loader = torch.utils.data.DataLoader(
    ds_push, batch_size=None, shuffle=False,
    num_workers=8,
    pin_memory=True)
test_loader = torch.utils.data.DataLoader(
    ds_test, batch_size=None, shuffle=False,
    num_workers=8,
    pin_memory=True)

# noinspection PyTypeChecker
log_writer.add_text('dataset_stats',
                    'training set size: {}, push set size: {}, test set size: {}'.format(
                        len(train_loader.dataset), len(train_push_loader.dataset), len(test_loader.dataset)))

# if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)


# train the model
print('Training started')


def write_mode(mode: TrainMode, log_writer: SummaryWriter, step: int):
    log_writer.add_scalar('mode/warm', int(mode == TrainMode.WARM), global_step=step)
    log_writer.add_scalar('mode/joint', int(mode == TrainMode.JOINT), global_step=step)
    log_writer.add_scalar('mode/push', int(mode == TrainMode.PUSH), global_step=step)
    log_writer.add_scalar('mode/last_only', int(mode == TrainMode.LAST_ONLY), global_step=step)


last_checkpoint = step
push_model_state = False

while True:
    print('step: {}, mode: {}, epoch: {}, iteration: {}'.format(step, mode.name, epoch, iteration))
    if mode == TrainMode.WARM:
        write_mode(TrainMode.WARM, log_writer, step)
        warm_only(model=ppnet)
        train(model=ppnet, dataloader=train_loader, optimizer=warm_optimizer,
              class_specific=config.class_specific, coefs=config.coefs, log_writer=log_writer, step=step)
        accu = test(model=ppnet, dataloader=test_loader,
                    class_specific=config.class_specific, log_writer=log_writer, step=step)
        push_model_state = False
        epoch += 1
        if epoch >= config.num_warm_epochs:
            mode = TrainMode.JOINT
    elif mode == TrainMode.JOINT:
        write_mode(TrainMode.JOINT, log_writer, step)
        joint(model=ppnet)
        train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer,
              class_specific=config.class_specific, coefs=config.coefs, log_writer=log_writer, step=step)
        joint_lr_scheduler.step()
        accu = test(model=ppnet, dataloader=test_loader,
                    class_specific=config.class_specific, log_writer=log_writer, step=step)
        push_model_state = False
        if epoch >= config.push_start and epoch in config.push_epochs:
            mode = TrainMode.PUSH
        else:
            epoch += 1
    elif mode == TrainMode.PUSH:
        write_mode(TrainMode.PUSH, log_writer, step)
        push_prototypes(
            train_push_loader,  # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet,
            class_specific=config.class_specific,
            preprocess_input_function=preprocess_input_function,  # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir,
            epoch_number=epoch,
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True)
        accu = test(model=ppnet, dataloader=test_loader,
                    class_specific=config.class_specific, log_writer=log_writer, step=step)
        push_model_state = True
        current_push_best_accu = 0.
        if config.prototype_activation_function != 'linear':
            mode = TrainMode.LAST_ONLY
            iteration = 0
        else:
            mode = TrainMode.JOINT
            epoch += 1
    elif mode == TrainMode.LAST_ONLY:
        print('iteration: \t{0}'.format(iteration))
        write_mode(TrainMode.LAST_ONLY, log_writer, step)
        last_only(model=ppnet)
        train(model=ppnet, dataloader=train_loader, optimizer=last_layer_optimizer,
              class_specific=config.class_specific, coefs=config.coefs, log_writer=log_writer, step=step)
        accu = test(model=ppnet, dataloader=test_loader,
                    class_specific=config.class_specific, log_writer=log_writer, step=step)
        iteration += 1
        push_model_state = True
        if iteration >= config.num_last_layer_iterations:
            iteration = None
            epoch += 1
            mode = TrainMode.JOINT
    else:
        raise NotImplementedError('unknown mode')
    if epoch >= config.num_train_epochs:
        break
    step += 1

    do_snapshot = False

    if last_checkpoint + CHECKPOINT_FREQUENCY_STEPS <= step:
        do_snapshot = True

    if accu > best_accu:
        do_snapshot = True
        best_model_path = os.path.join(model_dir, 'best')
        save_train_state(best_model_path, ppnet, other_state, step, mode, epoch, iteration, experiment_run_name,
                         best_accu, current_push_best_accu)
        best_accu = accu
        print('New best score: {}, saving snapshot'.format(best_accu))

    if push_model_state and accu > current_push_best_accu:
        do_snapshot = True
        push_best_model_path = os.path.join(model_dir, '{}.push.best'.format(epoch))
        save_train_state(push_best_model_path, ppnet, other_state, step, mode, epoch, iteration, experiment_run_name,
                         best_accu, current_push_best_accu)
        current_push_best_accu = accu
        print('Push {} best score: {}, saving snapshot'.format(epoch, best_accu))

    if do_snapshot:
        print('Saving checkpoint')
        save_train_state(checkpoint_file_prefix, ppnet, other_state, step, mode, epoch, iteration, experiment_run_name,
                         best_accu, current_push_best_accu)
        last_checkpoint = step

log_writer.close()
