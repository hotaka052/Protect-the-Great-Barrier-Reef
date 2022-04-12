from itertools import product
from math import sqrt

import torch

from .utils import match


class DefaultBox(object):
    """
    デフォルトのバウンディングボックスの作成
    """
    def __init__(self, cfg, device):
        """
        Args
            cfg: 設定をまとめたファイル
            device: "cpu" or "gpu"
        """
        super(DefaultBox, self).__init__()
        self.device = device

        self.image_size = cfg['input_size']
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg['feature_maps'])
        self.steps = cfg['steps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.aspect_ratios = cfg['aspect_ratios']

        # DefaultBoxを作成
        self.dbox_list = self.make_dbox()

    def make_dbox(self):
        """
        Return
            output: dboxの座標 torch.Size([8732, 4])
                    [center_x, center_y, width, height]
        """
        mean = []

        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]

                # 中心座標
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # 小さい正方形のdefault box
                s_k = self.min_sizes[k] / self.image_size
                # [center_x, center_y, width, height]
                mean += [cx, cy, s_k, s_k]

                # 大きい正方形のdefault box
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # その他のdefault box
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k * sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k / sqrt(ar)]

        output = torch.Tensor(mean).view(-1, 4)

        # 画像外にならないように調整
        output.clamp_(max=1, min=0)

        return output

    def get_targets(self, targets, jaccard_thresh=0.5):
        """
        dboxとtargetから教師データの作成

        Args:
            targets: 正解データ
            jaccard_thresh: positive boxを判定する際の閾値
        """

        num_batch = len(targets)
        num_dbox = self.dbox_list.size(0)

        # ================
        # 教師データの作成
        # ================

        # 教師データのひな型を準備
        conf_t = torch.LongTensor(num_batch, num_dbox).to(self.device)
        loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

        # targetとdboxから教師データを作成
        for idx in range(num_batch):
            truths = targets[idx][:, :-1].to(self.device)
            labels = targets[idx][:, -1].to(self.device)

            dbox = self.dbox_list.to(self.device)

            match(jaccard_thresh, truths, dbox, labels, loc_t, conf_t, idx)

        return loc_t, conf_t
