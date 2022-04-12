import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

from .utils import decode, nms


class Detect(Function):
    """
    推論時のボックス情報の出力
    """
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        """
        Args
            conf_thresh: 確信度の足きりライン
            top_k: 上位いくつを残すか
            nms_thresh: bboxがどれだけ被っていたら同じ物体とみなすか
        """
        self.sigmoid = nn.Sigmoid()
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh

    def __call__(self, loc_data: Tensor, conf_data: Tensor, dbox_list: Tensor):
        """
        Args
            loc_data: オフセット情報 torch.Size([batch_size, 8732, 4])
            conf_data: 確信度 torch.Size([batch_size, 8732, num_classes])
            dbox_list: デフォルトボックスの情報 torch.Size([8732, 4])
        Returns
            output: 推論したバウンディングボックスのリスト
        """
        num_batch = loc_data.size(0)
        num_classes = conf_data.size(2)

        # 確信度を正規化
        conf_data = self.sigmoid(conf_data)

        # 出力時のテンプレート型
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

        # [batch_size, 8732, num_classes] -> [batch_size, num_classes, 8732]
        conf_preds = conf_data.transpose(2, 1)

        for i in range(num_batch):
            # オフセット情報とdboxからBBoxを出力 [xmin, ymin, xmax, ymax]
            decoded_boxes = decode(loc_data[i], dbox_list)

            # 各ボックスの確信度 [num_classes, 8732]
            conf_scores = conf_preds[i].clone()

            # 閾値を超えたものを抜き出し
            conf_mask = conf_scores[0].gt(self.conf_thresh)
            scores = conf_scores[0][conf_mask]

            # 確信度が閾値を超えているものが無ければcontinue
            if scores.nelement() == 0:
                continue

            # conf_maskを使用してbbox情報を抜き出し
            loc_mask = conf_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[loc_mask].view(-1, 4)

            ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)

            # Non-Maximum Supressionを終えた結果
            output[i, 0, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output
