import pandas as pd
from sklearn.model_selection import train_test_split


def processing_csv(root_dir):
    """
    csvファイルの読み込みと加工
    """
    df = pd.read_csv(f'{root_dir}/train.csv')

    # 検出対象が写っていないのは削除
    df['annotations'] = df['annotations'].apply(eval)
    df['num_bbox'] = df['annotations'].apply(lambda x: len(x))
    df = df[df['num_bbox'] != 0].reset_index(drop=True)

    # dfに画像へのパスを追加
    df['img_path'] = f'{root_dir}/train_images/video_' + df.video_id.astype(str) + '/' + df.video_frame.astype(str) + '.jpg'

    # ラベルの追加
    df['labels'] = df['annotations'].apply(lambda x: [0] * len(x))

    return train_test_split(df, random_state=71)
