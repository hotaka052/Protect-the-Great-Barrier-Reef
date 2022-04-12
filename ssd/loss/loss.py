import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBoxLoss(nn.Module):
    def __init__(self, neg_pos, device):
        super(MultiBoxLoss, self).__init__()
        self.negpos_ratio = neg_pos
        self.device = device

    def forward(self, loc_data, conf_data, loc_t, conf_t):
        # positive dboxのマスクを作成
        pos_mask = conf_t > 0

        # オフセット情報に関するlossを計算
        loss_l = self._predict_loss_loc(loc_data, loc_t, pos_mask)

        # クラス予測に関するlossの計算
        loss_c = self._predict_loss_conf(conf_data, conf_t, pos_mask)

        # 最後にlossを割り算
        num_pos = pos_mask.long().sum(1, keepdim=True)
        N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c

    def _predict_loss_loc(self, loc_data, loc_t, pos_mask):
        """
        オフセット情報に関する損失の計算

        Args:
            loc_data: ネットワークが推測したオフセット情報
            loc_t: 正しいオフセット情報
            pos_mask: positive mask
        Returns:
            loss_l: オフセット情報に関するloss
        """
        # loc_dataに合わせて拡大
        pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

        # 推測データと教師データから抜き出し
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        
        # オフセット情報に関するlossを計算
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='mean')

        return loss_l

    def _predict_loss_conf(self, conf_data, conf_t, pos_mask):
        """
        クラスに関する損失の計算
        1. negative dboxに分類されたdboxからlossの計算に使うdboxを抽出
        2. positive dboxと選ばれたnegative dboxを合わせてクラス予測に関するlossを計算

        Args:
            conf_data: ネットワークが推測したクラスラベル
            conf_t: 正しいクラスラベル
            pos_mask: positive mask
        Returns:
            loss_c: クラスラベルに関するloss
        """

        num_batch = conf_data.size(0)
        num_dbox = conf_data.size(1)

        batch_conf = conf_data.view(-1)

        # positive box, negative boxに関わらずクラス予測に関するlossの計算
        loss_c = F.binary_cross_entropy_with_logits(
            batch_conf, conf_t.view(-1).float(), reduction='none')

        # positive dboxに関するlossは0に
        loss_c = loss_c.view(num_batch, -1)
        loss_c[pos_mask] = 0

        # lossを降順にソート
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # positive dboxの数からnegative boxの数を算出
        num_pos = pos_mask.long().sum(1, keepdim=True)
        num_neg = torch.clamp(num_pos * self.negpos_ratio, max=num_dbox)

        # negative dboxのマスクを作成
        neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # マスクの形をconf_dataに合わせる
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # conf_dataからlossの計算に使用するデータを抽出
        conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1)
        # conf_tからlossの計算に使用するデータを抽出
        conf_t_hnm = conf_t[(pos_mask+neg_mask).gt(0)]

        # クラス予測に関するlossの計算
        loss_c = F.binary_cross_entropy_with_logits(conf_hnm, conf_t_hnm.float(), reduction='mean')

        return loss_c
