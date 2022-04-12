import torch


def point_form(boxes):
    """
    デフォルトボックスを (xmin, ymin, xmax, ymax)に変換

    Args
        boxes: デフォルトボックス torch.Size([8732, 4])
        [center_x, center_y, width, height]
    Return
        [xmin, ymin, xmax, ymax]
    """
    return torch.cat((boxes[:, :2] - (boxes[:, 2:]/2),  # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,
                     boxes[:, 2:] - boxes[:, :2], 1)


def intersect(box_a, box_b):
    """
    重なっている部分の面積を計算

    Args:
        box_a: 正解のbbox torch.Size([ラベル数, 4])
        box_b: デフォルトボックス torch.Size([8732, 4])
        [xmin, ymin, xmax, ymax]
    Return
        重なっている部分の面積の配列 torch.Size([ラベル数, 8732])
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
    2つのバウンディングボックスの重なり具合を計算

    Args
        box_a: 正解のbbox torch.Size([ラベル数, 4])
        box_b: デフォルトボックス torch.Size([8732, 4])
        [xmin, ymin, xmax, ymax]
    Return
        重なり具合の配列 torch.Size([ラベル数, 8732])
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def match(threshold, truths, priors, labels, loc_t, conf_t, idx):
    """
    教師データの作成

    Args:
        threshold: Positive boxと判定する閾値
        truths: 正解のオフセット情報
        priors: デフォルトボックスのリスト
        labels: 正解のラベル情報
        loc_t: 各dboxのオフセット情報のひな型
        conf_t: 各dboxのラベル情報のひな型
        idx: バッチの番号
    Returns:
        1) loc_t: 各dboxのオフセット情報 torch.Size([batch_size, 8732, 4])
        2) conf_t: 各dboxのラベル情報 torch.size([batch_size, 8732]) 
    """
    # bboxとdboxの重なり具合に関する配列 torch.Size([ラベル数, 8732])
    overlaps = jaccard(
        truths,
        point_form(priors)
    )

    # 各バウンディングボックスに対して最も適合していたデフォルトボックスを抜き出し
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # 各デフォルトボックスに対して最も適合していたバウンディングボックスの抜き出し
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    # 配列の整形
    best_prior_idx.squeeze_(1)  # torch.Size([ラベル数])
    best_prior_overlap.squeeze_(1)  # torch.Size([ラベル数])
    best_truth_idx.squeeze_(0)  # torch.Size([8732])
    best_truth_overlap.squeeze_(0)  # torch.Size([8732])

    # 各バウンディングボックスに対して最もoverlapの高いデフォルトボックスの
    # overlapを2に設定しておく
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    # 各dboxのラベルに対応する正しいbbox
    matches = truths[best_truth_idx]  # torch.Size([8732, 4])
    # 各dboxに振り分けられるラベル
    conf = labels[best_truth_idx] + 1  # torch.Size([8732])
    # overlapが閾値を下回っているものは背景ラベルに振り分け
    conf[best_truth_overlap < threshold] = 0
    # dboxと正しいbboxから教師データとなるオフセット情報を求める
    loc = encode(matches, priors)

    # 元の配列に求めた値を戻す
    loc_t[idx] = loc
    conf_t[idx] = conf


def encode(matched, priors):
    """
    dboxとbboxの変化率の計算

    Args:
        matched: 各dboxに対応した正しいbbox torch.Size([8732, 4]) [xmin, ymin, xmax, ymax]
        priors: dbox情報 [cx. cy, width, height]
    Returns:
        torch.cat([g_cxcy, g_wh], 1): [Δx, Δy, Δwidth, Δheight]
    """
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]

    return torch.cat([g_cxcy, g_wh], 1)
