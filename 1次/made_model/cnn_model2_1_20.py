# 必要なモジュールのインポート
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

# データセットクラス
class LineDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[idx, 2]
        label_mapping = {"left": 0, "center": 1}
        label = label_mapping[label]
        if self.transform:
            image = self.transform(image)
        return image, label

# モデル定義
class LineDetectionCNN(nn.Module):
    def __init__(self):
        super(LineDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 学習ループ
def train_model(model, train_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# 評価
def evaluate_model(model, train_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy}%")
    return accuracy

# メイン処理
def main():
    # パス設定
    csv_file = r"C:\Users\USER\Desktop\dataset\valid_dataset_labels.csv"
    image_dir = r"C:\Users\USER\Desktop\dataset\train"
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # データローダーの設定
    train_dataset = LineDataset(csv_file=csv_file, root_dir=image_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # モデル構築
    model = LineDetectionCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 学習
    train_model(model, train_loader, device, epochs=10)

    # 評価
    evaluate_model(model, train_loader, device)

    # モデル保存
    torch.save(model.state_dict(), "line_detection_cnn.pth")
    print("モデルを保存しました: line_detection_cnn.pth")

if __name__ == "__main__":
    main()
