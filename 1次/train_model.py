import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# カスタムデータセットの定義
class SteeringDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # フォルダ名からステアリング角度を取得
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

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64))

        if self.transform:
            image = self.transform(image)

        angle = torch.tensor(angle, dtype=torch.float32)
        return image, angle

# データセットの準備
root_dir = r"C:\Users\USER\Desktop\processed_images"  # データセットのパス
transform = transforms.ToTensor()
dataset = SteeringDataset(root_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# CNNモデルの定義
class SteeringModel(nn.Module):
    def __init__(self):
        super(SteeringModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# モデルの初期化
model = SteeringModel()

# 損失関数とオプティマイザ
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# トレーニング
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for images, angles in data_loader:
        outputs = model(images)
        loss = criterion(outputs.squeeze(), angles.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader):.4f}")

# 学習済みモデルの保存
torch.save(model.state_dict(), "steering_model.pth")
print("Model saved as 'steering_model.pth'")
