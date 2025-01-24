import os
import pandas as pd  # pandasをインポート
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# データセットの準備
def prepare_dataset(csv_file, image_dir, output_csv):
    df = pd.read_csv(csv_file)
    file_list = os.listdir(image_dir)
    valid_rows = df[df["filename"].isin(file_list)]
    if valid_rows.empty:
        raise ValueError("有効なデータがありません。CSVファイルまたは画像フォルダを確認してください。")
    valid_rows.to_csv(output_csv, index=False)
    print(f"有効なデータセットを保存しました: {output_csv}")
    return valid_rows

# メイン処理
def main():
    csv_file = r"C:\Users\USER\Desktop\dataset\dataset_labels.csv"
    image_dir = r"C:\Users\USER\Desktop\dataset\train"
    valid_csv = r"C:\Users\USER\Desktop\dataset\valid_dataset_labels.csv"

    # データセットの準備
    prepare_dataset(csv_file, image_dir, valid_csv)

if __name__ == "__main__":
    main()


