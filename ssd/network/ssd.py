import torch
import torch.nn as nn
import torch.nn.functional as F

from .module_list import ssd_layers, ssd_predict_layers, vgg16
from .modules import L2Norm
from .utils import weights_init


class SSD(nn.Module):
    def __init__(self, cfg):
        super(SSD, self).__init__()

        self.vgg = vgg16(pretrained=True)
        self.ssd_layers = ssd_layers(cfg['ssd_layers'], base="vgg")
        self.L2Norm = L2Norm()
        self.loc, self.conf = ssd_predict_layers(cfg['multibox'], cfg['bbox_num'])

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # vggの計算(conv4-3まで)
        for k in range(23):
            x = self.vgg[k](x)

        source1 = self.L2Norm(x)
        sources.append(source1)

        # 残り分を計算
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        # ssd_layersの計算
        for k, v in enumerate(self.ssd_layers):
            x = F.relu(v(x), inplace=True)
            # conv -> relu -> conv -> reluを計算したらsourcesに追加
            if k % 2 == 1:
                sources.append(x)

        # ssd_predict_layersの計算
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # [batch_size, channels, height, width] -> [batch_size, height, width, channels]
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # tensorの形の変形 [batch_size, -1]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # [batch_size, dbox_num, -1]
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, 1)

        return loc, conf


def build_ssd(cfg):
    """
    ssdモデルの作成と重みの初期化

    Args:
        cfg: モデルの設定が記載してある辞書
    """
    model = SSD(cfg=cfg)

    model.ssd_layers.apply(weights_init)
    model.loc.apply(weights_init)
    model.conf.apply(weights_init)

    return model
