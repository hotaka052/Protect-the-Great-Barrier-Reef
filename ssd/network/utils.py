import torch.nn as nn


def weights_init(m):
    """
    重みの初期化
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
