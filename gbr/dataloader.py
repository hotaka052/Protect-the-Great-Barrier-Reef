import torch
from torch.utils.data import DataLoader

from .dataset import GBRDataset
from .processing import processing_csv
from .transforms import transforms


def get_dataloader(root_dir: str, batch_size=8):
    """
    DataLoaderの作成
    """
    train, val = processing_csv(root_dir)

    train_dataset = GBRDataset(
        df=train,
        transform=transforms
    )

    val_dataset = GBRDataset(
        df=val,
        transform=transforms
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader


def collate_fn(batch):
    targets = []
    images = []
    for sample in batch:
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    images = torch.stack(images, dim=0)

    return images, targets
