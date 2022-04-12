import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_list import (dssd_layers, dssd_predict_layers, resnet101,
                          ssd_layers)
from .utils import weights_init


class DeconvSSD(nn.Module):
    def __init__(self, cfg):
        super(DeconvSSD, self).__init__()

        self.resnet = resnet101(cfg["pretrained"])
        self.ssd_layers = ssd_layers(cfg['ssd_layers'], base="resnet")
        self.dssd_layers = dssd_layers(cfg["dssd_layers"])
        self.predict_layers = dssd_predict_layers(cfg["bbox_num"])

    def forward(self, x):
        loc = list()
        conf = list()
        ssd_outputs = list()
        dssd_outputs = list()

        # resnetの計算
        for k, v in enumerate(self.resnet):
            x = v(x)
            if k in [5, 6]:
                ssd_outputs.append(x)

        # ssd_layersの計算
        for k, v in enumerate(self.ssd_layers):
            x = F.relu(v(x), inplace=True)
            # conv -> relu -> conv -> reluを計算したらsourcesに追加
            if k % 2 == 1:
                ssd_outputs.append(x)

        # dssd_layersの計算
        for i in range(len(ssd_outputs)):
            if i == 0:
                x = ssd_outputs[(i+1) * -1]
                dssd_outputs.append(x)
                # print(x.shape)
            else:
                layer = self.dssd_layers[i-1]
                x = layer(x, ssd_outputs[(i+1) * -1])
                dssd_outputs.insert(0, x)
                # print(x.shape)

        # predict_layersの計算
        for (x, layer) in zip(dssd_outputs, self.predict_layers):
            l, c = layer.forward(x)
            loc.append(l.permute(0, 2, 3, 1).contiguous())
            conf.append(c.permute(0, 2, 3, 1).contiguous())

        # tensorの形の変形 [batch_size, -1]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # [batch_size, dbox_num, -1]
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, 1)

        return loc, conf


def build_dssd(cfg):
    """
    ssdモデルの作成と重みの初期化

    Args:
        cfg: モデルの設定が記載してある辞書
    Returns:
        モデルインスタンス
    """
    model = DeconvSSD(cfg)

    model.ssd_layers.apply(weights_init)
    model.dssd_layers.apply(weights_init)
    model.predict_layers.apply(weights_init)

    return model
