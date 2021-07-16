import torch.nn as nn


class NoopModel(nn.Module):
    def forward(self, x):
        return x

    def conv_info(self):
        # resnet18
        return ([7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
                [2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1],
                [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def noop_features(pretrained=False, batch_norm=True, **kwargs):
    model = NoopModel()
    return model
