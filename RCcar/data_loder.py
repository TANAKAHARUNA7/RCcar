import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# **1. カスタムデータセットクラス**
class LaneDataset(Dataset):
    def __init__(self, csv_file, root_dir, split, transform=None):
        """
        Args:
            csv_file (str): CSVファイルのパス
            root_dir (str): 画像データセットのルートディレクトリ
            split (str): 'train', 'val', 'test' の指定
            transform (callable, optional): 画像の前処理
        """
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]  # 指定されたsplitでフィルタリング
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapping = {"left": 0, "center": 1, "right": 2}  # ラベルを数値化

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # サブフォルダとファイル名を組み合わせて画像パスを生成
        subfolder = self.data.iloc[idx, 1]  # label列 (left, center, right)
        filename = self.data.iloc[idx, 0]   # filename列
        img_path = os.path.join(self.root_dir, subfolder, filename)

        # 画像読み込みと前処理
        image = Image.open(img_path).convert("RGB")
        label = self.label_mapping[subfolder]
        if self.transform:
            image = self.transform(image)
        return image, label

# **2. CSVファイルの修正**
def clean_csv(csv_file, root_dir):
    print("存在しないファイルを確認中...")
    data = pd.read_csv(csv_file)
    valid_entries = []
    for _, row in data.iterrows():
        split = row['split']
        label = row['label']
        filename = row['filename']
        file_path = os.path.join(root_dir, split, label, filename)
        if os.path.exists(file_path):
            valid_entries.append(row)
        else:
            print(f"存在しないファイル: {file_path}")
    # 有効なデータのみを保持
    cleaned_data = pd.DataFrame(valid_entries)
    cleaned_csv = os.path.join(root_dir, "fixed_dataset_labels.csv")
    cleaned_data.to_csv(cleaned_csv, index=False)
    print(f"修正済みCSVが保存されました: {cleaned_csv}")
    return cleaned_csv

# **3. データローダーの確認関数**
def check_dataloader(loader, loader_name):
    print(f"=== {loader_name} データローダーの確認 ===")
    for images, labels in loader:
        print(f"画像のバッチサイズ: {images.shape}")
        print(f"ラベル: {labels}")
        break  # 最初のバッチのみ表示

# **4. メイン処理**
if __name__ == "__main__":
    # パス設定
    csv_file = r"C:\Users\USER\Desktop\dataset\dataset_labels.csv"
    root_dir = r"C:\Users\USER\Desktop\dataset"
    batch_size = 32

    # CSVの修正
    fixed_csv_file = clean_csv(csv_file, root_dir)

    # 前処理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # データセットとデータローダーの作成
    train_dataset = LaneDataset(fixed_csv_file, os.path.join(root_dir, "train"), "train", transform)
    val_dataset = LaneDataset(fixed_csv_file, os.path.join(root_dir, "val"), "val", transform)
    test_dataset = LaneDataset(fixed_csv_file, os.path.join(root_dir, "test"), "test", transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # データローダーの動作確認
    check_dataloader(train_loader, "Train")
    check_dataloader(val_loader, "Validation")
    check_dataloader(test_loader, "Test")


