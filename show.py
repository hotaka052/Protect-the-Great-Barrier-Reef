import argparse

from ssd import VisualizeBbox

parser = argparse.ArgumentParser(
    description = "parameter for mnist training"
)
parser.add_argument("--root_dir", default="/kaggle/input/tensorflow-great-barrier-reef", type=str,
                    help="ルートディレクトリへのパス")
parser.add_argument("--weight_path", type=str,
                    help="モデルの重みへのパス")
parser.add_argument("--threshold", default=0.5, type=float,
                    help="同一のものを検出していると判断する閾値")
parser.add_argument("--sample_num", default=6, type=int,
                    help="推論を出力したいデータの個数")
args = parser.parse_args()

import warnings

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    ssd = VisualizeBbox(root_dir=args.root_dir, threshold=args.threshold, weight_path=args.weight_path)
    ssd.show(args.sample_num)
