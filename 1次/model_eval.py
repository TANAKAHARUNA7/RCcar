import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import cv2
from PIL import Image

# --- 学習済みモデルをロード ---
from train_model import SteeringModel  # 正しいファイル名を指定

model = SteeringModel()  # モデルのインスタンス化
model.load_state_dict(torch.load("steering_model.pth"))  # モデルの重みをロード
model.eval()  # 評価モードに設定

# --- データセットの作成 ---
class SteeringDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            if '-' in folder_name:
                angle = int(folder_name.split('-')[0])
            else:
                angle = int(folder_name)

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                self.data.append(image_path)
                self.labels.append(angle)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        angle = self.labels[idx]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((64, 64), Image.Resampling.LANCZOS)

        if self.transform:
            image = self.transform(image)

        angle = torch.tensor(angle, dtype=torch.float32)
        return image, angle

# --- データローダーの作成 ---
root_dir = r"C:\Users\USER\Desktop\processed_images"  # データセットのパス
transform = transforms.ToTensor()
dataset = SteeringDataset(root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# --- 損失関数 ---
criterion = nn.MSELoss()

# --- 評価プロセス ---
test_loss = 0
total_samples = 0

with torch.no_grad():  # 勾配計算を無効化
    for images, angles in data_loader:
        outputs = model(images)  # 推論
        loss = criterion(outputs.squeeze(), angles.float())  # 損失計算
        test_loss += loss.item()
        total_samples += len(images)

# 平均損失を表示
print(f"Test Loss: {test_loss / len(data_loader):.4f}")
print(f"Average Loss per Sample: {test_loss / total_samples:.4f}")


