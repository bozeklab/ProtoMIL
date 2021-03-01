import os
import random
import subprocess
from glob import glob
from typing import Dict, Any, Optional, Tuple

import numpy
import torch

from train_and_test import TrainMode


def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''

    if accu >= target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu * 100)))


def save_train_state(file_path_prefix: str, model: torch.nn.Module, things_with_state: Dict[str, Any],
                     step: int, mode: TrainMode, epoch: int, iteration: Optional[int], experiment_run_name: str,
                     best_accu: float, current_push_best_accu: Optional[float], current_accu: float):
    file_path = '{}.{}.{:.2f}.pck'.format(file_path_prefix, step, current_accu * 100)
    # save and atomic replace
    new_file_path = file_path + '.new'
    try:
        torch.save({
            'model': model.state_dict(),
            'states': {name: obj.state_dict() for name, obj in things_with_state.items()},
            'step': step,
            'mode': mode,
            'epoch': epoch,
            'iteration': iteration,
            'experiment_run_name': experiment_run_name,
            'best_accu': best_accu,
            'current_push_best_accu': current_push_best_accu,
            'torch_random': torch.random.get_rng_state(),
            'numpy_random': numpy.random.get_state(),
            'python_random': random.getstate(),
        }, new_file_path)
        # atomic on POSIX
        os.replace(new_file_path, file_path)
        files = get_state_path_for_prefix(file_path_prefix)
        [os.remove(f) for f in files if f != file_path]
    finally:
        if os.path.exists(new_file_path):
            os.remove(new_file_path)


def get_state_path_for_prefix(file_path_prefix: str):
    files = glob(file_path_prefix + '*')
    return files


def load_train_state(file_path: str, model: torch.nn.Module, things_with_state: Dict[str, Any],
                     restore_random_state: bool = True) -> \
        Tuple[int, TrainMode, int, Optional[int], str, float, Optional[float]]:
    data = torch.load(file_path)
    model.load_state_dict(data['model'])
    for name, obj in things_with_state.items():
        obj.load_state_dict(data['states'][name])
    if restore_random_state and 'torch_random' in data:
        torch.random.set_rng_state(data['torch_random'])
        numpy.random.set_state(data['numpy_random'])
        random.setstate(data['python_random'])
    return (data['step'],
            data['mode'],
            data['epoch'],
            data['iteration'],
            data['experiment_run_name'],
            data.get('best_accu', 0.),
            data.get('current_push_best_accu', None))


def load_model_from_train_state(file_path: str, model: torch.nn.Module):
    data = torch.load(file_path)
    if 'model' in data:
        model.load_state_dict(data['model'])
    else:
        model.load_state_dict(data)


def _git_command(*args: str):
    subprocess.check_output(['git', '--git-dir=experiments', '--work-tree=.'] + list(args))


def _init_git_if_required():
    _git_command('init', '-q')


def snapshot_code(experiment_run_name):
    _init_git_if_required()
    _git_command('add', '*.py')
    _git_command('commit', '--allow-empty', '-m', experiment_run_name, '-q')
