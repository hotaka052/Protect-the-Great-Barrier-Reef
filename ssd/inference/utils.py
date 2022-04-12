import torch


def decode(loc: torch.Tensor, dbox_list: torch.Tensor) -> torch.Tensor:
    """Dboxと出力されたオフセット情報からBBoxを作成
    Args
        loc: モデルが出力したオフセット情報 [Δcx, Δcy, Δwidth, Δheight]
        dbox_list: Dboxの情報 [cx, cy, width, height]
    Return
        boxes: [xmin, ymin, xmax, ymax]
    """

    # オフセットとdboxの情報からbboxをまとめて計算
    boxes = torch.cat((dbox_list[:, :2] + 0.1 * loc[:, :2] * dbox_list[:, 2:],
                       dbox_list[:, 2:] * torch.exp(0.2 * loc[:, 2:])), dim=1)

    # bboxの情報を[cx, cy, width, height] -> [xmin, ymin, xmax, ymax]に変換
    boxes[:, :2] -= boxes[:, 2:] / 2  # [xmin, ymin] = [cx, cy] - [width / 2, height / 2]
    boxes[:, 2:] += boxes[:, :2]  # [xmax, ymax] = [width, height] + [xmin, ymin]

    return boxes


def nms(boxes: torch.Tensor, scores: torch.Tensor, overlap=0.45, top_k=200):
    """検出した物体が被っているbboxの削除
    Args
        boxes: bbox情報 torch.Size([閾値を越えたBox数, 4])
        scores: bboxの確信度 torch.Size([閾値を越えたBox数])
        overlap: どれだけ被っていたら削除するか
        top_k: bboxをいくつまで残すか
    Returns
        keep: 返却するindex
        count: 残すbboxの数
    """
    count = 0
    keep = scores.new_zeros(scores.size(0)).long()

    # 各BBoxの面積を計算
    x1 = boxes[:, 0]  # xmin
    y1 = boxes[:, 1]  # ymin
    x2 = boxes[:, 2]  # xmax
    y2 = boxes[:, 3]  # ymax
    area = torch.mul(x2 - x1, y2 - y1)

    # scoreを昇順に並び替える
    _, idx = scores.sort(0)

    # top_k個でスライス
    idx = idx[-top_k:]

    while idx.numel() > 0:
        i = idx[-1]  # confが最大のindexを取り出す

        # indexをkeepに追加
        keep[count] = i
        count += 1

        if idx.size(0) == 1:
            break

        # idxを一つ減らす
        idx = idx[:-1]

        # ===============================
        # keepに格納したbboxと被りの大きいbboxのindexを抽出して除去
        # ===============================

        # 現在取得しているindex以外のbboxの情報を取得
        xx1 = torch.index_select(x1, 0, idx)  # xmin
        yy1 = torch.index_select(y1, 0, idx)  # ymin
        xx2 = torch.index_select(x2, 0, idx)  # xmax
        yy2 = torch.index_select(y2, 0, idx)  # yma

        # 重なっている部分の座標を取得
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])

        # 重なっている部分のwidth, heightを取得
        w = xx2 - xx1
        h = yy2 - yy1

        # 調整
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # 重なっている部分の面積
        inter = w * h

        # 現在取得しているindex以外のbboxの面積
        rem_areas = torch.index_select(area, 0, idx)
        # 全体の面積
        union = (rem_areas - inter) + area[i]
        # 重なっている割合を取得
        IoU = inter / union

        # 閾値を超えていたindexは削除
        idx = idx[IoU.le(overlap)]

    return keep, count
