from torch import nn


class Lenet5_features(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            # nn.ReLU(),
            nn.Sigmoid(),
            # nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.feature_extractor(x)

    def conv_info(self):
        kernel_sizes = [5, 2, 5]
        strides = [1, 2, 1]
        paddings = [0, 0, 0]
        return kernel_sizes, strides, paddings


def lenet5_features(pretrained=False, **kwargs):
    return Lenet5_features()
