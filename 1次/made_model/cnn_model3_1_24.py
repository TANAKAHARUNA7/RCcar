import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np



# データセットのルートパス
data_root = r"C:\Users\USER\Desktop\processed_images"

# データ変換（正規化 + 64x64の画像用）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 平均=0.5, 標準偏差=0.5で正規化
])

# データセットの読み込み
train_dataset = datasets.ImageFolder(root=data_root, transform=transform)

# DataLoaderでデータセットを準備
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# クラス（フォルダ名）を取得
classes = train_dataset.classes
print(f"Classes: {classes}")

# サンプル画像を表示（確認用）
def imshow(img):
    img = img / 2 + 0.5  # デノーマライズ
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

data_iter = iter(train_loader)
images, labels = next(data_iter)
imshow(torchvision.utils.make_grid(images))
print("Labels:", labels)

# CNNモデルの定義
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)  # 入力サイズは (64 x 16 x 16)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)  # フラット化
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# モデルの初期化
num_classes = len(classes)
model = SimpleCNN(num_classes)

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# トレーニングループ
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        # データをGPUに移動（必要であれば）
        inputs, labels = inputs, labels

        # 勾配の初期化
        optimizer.zero_grad()

        # 順伝播 + 逆伝播 + 重み更新
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 損失を表示
        running_loss += loss.item()
        if i % 10 == 9:  # 10バッチごとにログを出力
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

print("Finished Training")

# モデルの保存
torch.save(model.state_dict(), "made_model.pth")
print("Model saved as model.pth")
