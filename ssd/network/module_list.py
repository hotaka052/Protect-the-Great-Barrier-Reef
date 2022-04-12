import torch.nn as nn
from torchvision import models

from .modules import DeconvModule, PredModule


def vgg16(pretrained) -> nn.ModuleList:
    """vgg16の準備
    Args:
        pretrained: 学習済み重みを適用するか
    """
    base = models.vgg16(pretrained=pretrained)
    layers = list(base.features)[:-1]

    layers += extras(512)

    return nn.ModuleList(layers)


def resnet101(pretrained) -> nn.ModuleList:
    """resnet101の準備
    Args:
        pretrained: 学習済み重みを適用するか
    """
    base = models.resnet101(pretrained=pretrained)
    layers = list(base.children())[:-2]

    layers += extras(2048)

    return nn.ModuleList(layers)


def extras(in_channels) -> list:
    """追加分のレイヤー
    Args:
        in_channels: チャンネル数
    """
    layers = []

    layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
    layers += [nn.Conv2d(in_channels, 1024, kernel_size=3, padding=6, dilation=6)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(1024, 1024, kernel_size=1)]
    layers += [nn.ReLU(inplace=True)]

    return layers


def ssd_layers(cfg, base) -> nn.ModuleList:
    """ssdの出力に使うレイヤー
    Args
        cfg: 層のチャンネル数
    """

    layers = []
    in_channels = 1024

    if "vgg" in base:
        stride = 2
    elif "resnet" in base:
        stride = 1

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=1)]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=3, stride=stride, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=1)]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=3, stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=1)]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=3)]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=1)]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=3)]

    return nn.ModuleList(layers)


def dssd_layers(cfg):
    """dssdの追加レイヤー
    Args:
        cfg["kernel_size_list"]: レイヤーのカーネルサイズ
        cfg["channel_list]: レイヤーの入力チャンネル数
    """
    layers = []

    kernel_size_list = cfg["kernel_size_list"]
    channel_list = cfg["channel_list"]

    for i in range(len(kernel_size_list)):
        layers += [DeconvModule(channel_list[i], kernel_size_list[i])]

    return nn.ModuleList(layers)


def dssd_predict_layers(bbox_num):
    """dssdの出力レイヤー
        bbox_num: 出力するBBoxの個数
    """
    layers = []

    for i in range(len(bbox_num)):
        layers += [PredModule(bbox_num[i])]

    return nn.ModuleList(layers)


def ssd_predict_layers(cfg, bbox_num):
    """ssdの出力レイヤー
    Args
        cfg: チャンネル数
        bbox_num: 出力するBBoxの個数
    """

    loc_layers = []
    conf_layers = []

    for i in range(len(cfg)):
        loc_layers += [nn.Conv2d(cfg[i], bbox_num[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(cfg[i], bbox_num[i], kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)
