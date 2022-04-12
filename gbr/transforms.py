import albumentations as A
from albumentations.pytorch import ToTensorV2

transforms = A.Compose([
    A.Resize(width=320, height=320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))

train_transforms = A.Compose([
    A.RandomSizedBBoxSafeCrop(width=320, height=320),
    A.GaussNoise(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))

val_transforms = A.Compose([
    A.Resize(width=320, height=320),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
], bbox_params=A.BboxParams(format='albumentations', label_fields=['labels']))
