import os
from typing import Dict, Any, Optional, Tuple

import torch

from train_and_test import TrainMode


def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''

    if accu >= target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))


def save_train_state(file_path: str, model: torch.nn.Module, things_with_state: Dict[str, Any],
                     step: int, mode: TrainMode, epoch: int, iteration: Optional[int], experiment_run_name: str,
                     best_accu: float, current_push_best_accu: Optional[float]):
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
        }, new_file_path)
        # atomic on POSIX
        os.replace(new_file_path, file_path)
    finally:
        if os.path.exists(new_file_path):
            os.remove(new_file_path)


def load_train_state(file_path: str, model: torch.nn.Module, things_with_state: Dict[str, Any]) -> \
        Tuple[int, TrainMode, int, Optional[int], str, float, Optional[float]]:
    data = torch.load(file_path)
    model.load_state_dict(data['model'])
    for name, obj in things_with_state.items():
        obj.load_state_dict(data['states'][name])
    return (data['step'],
            data['mode'],
            data['epoch'],
            data['iteration'],
            data['experiment_run_name'],
            data.get('best_accu', 0.),
            data.get('current_push_best_accu', None))


def load_model_from_train_state(file_path: str, model: torch.nn.Module):
    data = torch.load(file_path)
    model.load_state_dict(data['model'])
