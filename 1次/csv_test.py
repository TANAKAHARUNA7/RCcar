import os
import pandas as pd
import random

# 元フォルダのパス
image_dir = r"C:\Users\USER\Desktop\RCcar_image"
output_csv = r"C:\Users\USER\Desktop\dataset_labels.csv"

# ラベル設定関数（ランダムにleft, right, centerを割り当てる例）
def assign_label(filename):
    if "left" in filename:
        return "left"
    elif "right" in filename:
        return "right"
    else:
        return random.choice(["left", "right", "center"])  # 仮のラベル

# 画像リストの取得
data = []
for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        label = assign_label(filename)
        split = random.choices(["train", "val", "test"], weights=[70, 15, 15])[0]  # 70:15:15に分割
        data.append([filename, label, split])

# CSV作成
df = pd.DataFrame(data, columns=["filename", "label", "split"])
df.to_csv(output_csv, index=False)
print("dataset_labels.csvが作成されました！")
