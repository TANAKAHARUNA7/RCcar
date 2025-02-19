import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
# Jetson Nano用データセットのパス（適宜修正）
dataset_path = r"C:\Users\USER\RCcar\dataset3"

# フォルダ名ごとの角度マッピング
angle_mapping = {
    "60-74": 70,
    "76-89": 80,
    "90": 90,
    "91-105": 100,
    "106-120": 110
}

# 画像の前処理（リサイズ & 正規化）
transform = transforms.Compose([
    transforms.Resize((66, 66)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# データセットクラス
class LaneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for folder, angle in angle_mapping.items():
            folder_path = os.path.join(root_dir, folder)
            if os.path.exists(folder_path):
                for img_name in os.listdir(folder_path):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        self.image_paths.append(os.path.join(folder_path, img_name))
                        self.labels.append(angle)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

# データローダーを作成
dataset = LaneDataset(root_dir=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"✅ データセット準備完了！画像数: {len(dataset)} 枚")

# CNNモデルの定義
class LaneFollowerCNN(nn.Module):
    def __init__(self):
        super(LaneFollowerCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)  # 1つの角度を回帰

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 予測した角度を返す

# モデルのインスタンス化
model = LaneFollowerCNN()
print("✅ モデルの構築が完了しました！")

# 損失関数 & 最適化アルゴリズム
criterion = nn.MSELoss()  # 回帰なので MSELoss を使用
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20  # 学習回数

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        labels = labels.view(-1, 1)  # (batch_size, 1) に変換

        # 順伝播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

print("✅ モデルのトレーニングが完了しました！")

torch.save(model.state_dict(), "lane_follower_cnn.pth")
print("✅ モデルの保存が完了しました！")
