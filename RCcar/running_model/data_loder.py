import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# カスタムデータセットの定義
class SteeringDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # データセットの作成
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # フォルダ名からステアリング角度を取得
            if '-' in folder_name:
                angle = int(folder_name.split('-')[0])  # 例: "60-74" → 60度
            else:
                angle = int(folder_name)  # 例: "90" → 90度

            # フォルダ内の画像をリストに追加
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                self.data.append(image_path)
                self.labels.append(angle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        angle = self.labels[idx]

        # 画像を読み込み
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR → RGB
        image = cv2.resize(image, (64, 64))  # サイズを64x64に変更

        if self.transform:
            image = self.transform(image)

        # ステアリング角度をテンソル化
        angle = torch.tensor(angle, dtype=torch.float32)
        return image, angle


# データセットの読み込み
root_dir = root_dir = r"C:\Users\USER\Desktop\processed_images"
transform = transforms.ToTensor()  # 画像をテンソルに変換
dataset = SteeringDataset(root_dir, transform=transform)

# データローダーの作成
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# データセットのサンプルを確認
image, angle = dataset[0]  # 最初の画像とラベルを取得
print(f"Image shape: {image.shape}, Steering Angle: {angle}")

# 画像を表示（確認用）
plt.imshow(image.permute(1, 2, 0))  # チャンネルを移動して表示
plt.title(f"Steering Angle: {angle.item()}")
plt.show()
