import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# **カスタムデータセットクラス**
class LaneDataset(Dataset):
    def __init__(self, csv_file, root_dir, split, transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data['split'] == split]
        self.root_dir = root_dir
        self.transform = transform
        self.label_mapping = {"left": 0, "center": 1, "right": 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subfolder = self.data.iloc[idx, 1]  # label列
        filename = self.data.iloc[idx, 0]   # filename列
        img_path = os.path.join(self.root_dir, subfolder, filename)

        image = Image.open(img_path).convert("RGB")
        label = self.label_mapping[subfolder]

        if self.transform:
            image = self.transform(image)
        return image, label

# **CNNモデルの定義**
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 出力: 64x64

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 出力: 32x32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 出力: 16x16
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 出力クラス: left, center, right
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# **前処理とデータローダーの作成**
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# パス設定
csv_file = r"C:\Users\USER\Desktop\dataset\fixed_dataset_labels.csv"
root_dir_train = r"C:\Users\USER\Desktop\dataset\train"
root_dir_val = r"C:\Users\USER\Desktop\dataset\val"
root_dir_test = r"C:\Users\USER\Desktop\dataset\test"

train_dataset = LaneDataset(csv_file, root_dir_train, "train", transform)
val_dataset = LaneDataset(csv_file, root_dir_val, "val", transform)
test_dataset = LaneDataset(csv_file, root_dir_test, "test", transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# **学習の設定**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **学習ループ**
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # 順伝播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 勾配の計算と更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # **Validation**
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

# **Testデータで最終評価**
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")
