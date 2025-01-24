import os
import pandas as pd

# パス設定
csv_file = r"C:\Users\USER\Desktop\dataset\dataset_labels.csv"
root_dir = r"C:\Users\USER\Desktop\dataset\train"

# 1. CSV読み込み
df = pd.read_csv(csv_file)

# 2. ファイル存在確認と不足ファイルの除外
valid_rows = []
missing_files = []

for _, row in df.iterrows():
    file_path = os.path.join(root_dir, row["filename"])
    if os.path.exists(file_path):
        valid_rows.append(row)
    else:
        missing_files.append(file_path)

# 不足しているファイルを表示
if missing_files:
    print("以下のファイルが見つかりませんでした:")
    for file in missing_files:
        print(file)

# 有効なデータのみを新しいCSVファイルに保存
valid_df = pd.DataFrame(valid_rows)
valid_csv_file = r"C:\Users\USER\Desktop\dataset\valid_dataset_labels.csv"
valid_df.to_csv(valid_csv_file, index=False)
print(f"有効なデータで更新されたCSVファイルを保存しました: {valid_csv_file}")

# 3. データセット準備用のクラス
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LineDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[idx, 2]
        label_mapping = {"left": 0, "center": 1, "right": 2}
        label = label_mapping[label]
        if self.transform:
            image = self.transform(image)
        return image, label

# データローダーのテスト
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = LineDataset(
    csv_file=valid_csv_file,
    root_dir=root_dir,
    transform=transform
)

print(f"データセットのサイズ: {len(dataset)}")
