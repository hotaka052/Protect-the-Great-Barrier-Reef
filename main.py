import argparse
import warnings

from gbr import get_dataloader
from ssd import TrainModel

parser = argparse.ArgumentParser(
    description="parameter for ssd training"
)
parser.add_argument("--root_dir", default="/kaggle/input/tensorflow-great-barrier-reef", type=str,
                    help="ルートディレクトリへのパス")
parser.add_argument("--epochs", default=300, type=int,
                    help="学習を何エポック回すか")
parser.add_argument("--es_patience", default=20, type=int,
                    help="何エポックスコアの改善が無かった時に学習を止めるか")
parser.add_argument("--lr", default=1e-2, type=float,
                    help="学習率")
parser.add_argument("--seed", default=71, type=int,
                    help='シード値')
parser.add_argument("--vgg_weight_path", default=None, type=str,
                    help="vgg16の重みのパス")

args = parser.parse_args()

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    train_dataloader, val_dataloader = get_dataloader(args.root_dir)

    trainer = TrainModel(
        epochs=args.epochs, es_patience=args.es_patience,
        train_dl=train_dataloader, val_dl=val_dataloader,
        lr=args.lr, seed=args.seed, vgg_weight_path=args.vgg_weight_path
    )

    trainer.train()
