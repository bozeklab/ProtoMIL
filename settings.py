from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class Settings:
    base_architecture: str
    img_size: int
    prototype_shape: Tuple[int, int, int, int]

    joint_optimizer_lrs: dict
    joint_lr_step_size: int
    warm_optimizer_lrs: dict
    last_layer_optimizer_lr: dict

    num_train_epochs: int
    num_warm_epochs: int
    num_last_layer_iterations: int
    push_start: int
    push_epochs: List[int]

    num_classes: int = 2
    coef_crs_ent: float = 1
    coef_clst: float = 0.8
    coef_sep: float = -0.08
    coef_l1: float = 1e-4

    prototype_activation_function: str = 'log'
    add_on_layers_type: str = 'regular'
    class_specific: bool = True

    @property
    def coefs(self):
        return {
            'crs_ent': self.coef_crs_ent,
            'clst': self.coef_clst,
            'sep': self.coef_sep,
            'l1': self.coef_l1,
        }


MNIST_SETTINGS = Settings(
    base_architecture='resnet18_small',
    img_size=28,
    prototype_shape=(20, 512, 2, 2),
    joint_optimizer_lrs={
        'features': 1e-4,
        'add_on_layers': 3e-3,
        'prototype_vectors': 3e-3,
    },
    joint_lr_step_size=5,
    warm_optimizer_lrs={
        'add_on_layers': 3e-3,
        'prototype_vectors': 3e-3,
    },
    last_layer_optimizer_lr={
        'attention': 1e-3,
        'last_layer': 1e-4,
    },
    num_train_epochs=101,
    num_warm_epochs=5,
    num_last_layer_iterations=20,
    push_start=10,
    push_epochs=[i for i in range(1000) if i % 10 == 0]
)

COLON_CANCER_SETTINGS = Settings(
    base_architecture='resnet18_small',
    img_size=27,
    prototype_shape=(10, 128, 2, 2),
    joint_optimizer_lrs={
        'features': 1e-4,
        'add_on_layers': 3e-3,
        'prototype_vectors': 3e-3,
    },
    joint_lr_step_size=5,
    warm_optimizer_lrs={
        'add_on_layers': 3e-3,
        'prototype_vectors': 3e-3,
    },
    last_layer_optimizer_lr={
        'attention': 1e-3,
        'last_layer': 1e-4,
    },
    num_train_epochs=101,
    num_warm_epochs=5,
    num_last_layer_iterations=20,
    push_start=10,
    push_epochs=[i for i in range(1000) if i % 10 == 0]
)
