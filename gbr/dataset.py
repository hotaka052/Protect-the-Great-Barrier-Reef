import cv2
import numpy as np
from torch.utils.data import Dataset


class GBRDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # 画像読み込み
        image = cv2.imread(row['img_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # bboxの取得
        height, width, _ = image.shape
        bboxes = get_bbox(row['annotations'], height, width)

        # transformerで変換
        transformed = self.transform(
            image=image, bboxes=bboxes, labels=row['labels'])
        image = transformed['image']
        bboxes = transformed['bboxes']
        labels = transformed['labels']

        targets = np.hstack((bboxes, np.expand_dims(labels, axis=1)))

        return image, targets


def get_bbox(anno_info, height, width):
    """
    [x, y, height, width] -> [xmin, ymin, xmax, ymax]
    """
    bbox = []
    for obj in anno_info:
        bbox_list = []
        bbox_list.append(obj['x'] / width)  # xmin
        bbox_list.append(obj['y'] / height)  # ymin
        bbox_list.append(min(1.0, (obj['x'] + obj['width']) / width))  # xmax
        bbox_list.append(min(1.0, (obj['y'] + obj['height']) / height))  # ymax

        bbox.append(bbox_list)

    return bbox
