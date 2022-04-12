import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from albumentations.pytorch import ToTensorV2

from .inference import Detect
from .network import build_dssd
from .training_data import DefaultBox


class VisualizeBbox:
    def __init__(self, weight_path: str, cfg_path: str, threshold=0.5, root_dir="/kaggle/input/tensorflow-great-barrier-reef"):
        with open(cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.threshold = threshold
        self.model = self._set_model(weight_path, cfg["ssd"])
        self.df = self._processing_data(root_dir)

        self.detect = Detect()
        device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.dbox_list = DefaultBox(cfg=cfg["dbox"], device=device).dbox_list

        # tranform
        self.transform = A.Compose([
            A.Resize(width=320, height=320),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def show(self, sample_num=6, nrows=2, ncols=3):

        df = self.df.sample(sample_num)

        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(60, 20))

        for i in range(len(df)):
            _r = i // ncols
            _c = i % ncols
            img, bboxes = self._get_bbox(df.iloc[i])
            self._show_bbox(img, bboxes, ax, _r, _c)

    def _get_bbox(self, row):
        # 画像の読み込み
        img = cv2.imread(row['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 変換
        transformed = self.transform(image=img)
        img_transformed = transformed['image']
        img_transformed = img_transformed.unsqueeze(0)

        # 推論
        self.model.eval()
        loc, conf = self.model(img_transformed)
        detections = self.detect(loc, conf, self.dbox_list)
        detections = detections.cpu().detach().numpy()

        # 物体を発見したbboxを抽出
        bboxes = []

        find_index = np.where(detections[:, :, :, 0] >= self.threshold)
        detections = detections[find_index]

        height, width, _ = img.shape

        for i in range(len(detections)):
            bboxes.append(detections[i][1:]*[width, height, width, height])

        return img, bboxes

    def _show_bbox(self, img, bboxes, ax, _r, _c):
        """
        画像とbboxの描画
        """
        height, width, _ = img.shape

        ax[_r, _c].imshow(img)

        for bbox in bboxes:
            # 枠の座標
            xy = (bbox[0], bbox[1])
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # 長方形を描画する
            ax[_r, _c].add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor="red", linewidth=2))

    def _processing_data(self, root_dir):
        """
        データの準備
        """
        df = pd.read_csv(f"{root_dir}/train.csv")
        df['annotations'] = df['annotations'].apply(eval)
        df['num_bbox'] = df['annotations'].apply(lambda x: len(x))
        df = df[df['num_bbox'] > 5].reset_index(drop=True)
        df['img_path'] = f'{root_dir}/train_images/video_' + \
            df.video_id.astype(str) + '/' + df.video_frame.astype(str) + '.jpg'

        return df

    def _set_model(self, weight_path: str, cfg: dict):
        """
        モデルの準備
        """
        model = build_dssd(cfg=cfg)
        weights = torch.load(weight_path, map_location={'cuda:0': 'cpu'})
        model.load_state_dict(weights, strict=False)

        return model
