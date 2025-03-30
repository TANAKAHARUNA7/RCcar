import os
import shutil
import pandas as pd

# パス設定
image_dir = r"C:\Users\USER\Desktop\RCcar_img"  # 元のイメージがあるフォルダ
output_dir = r"C:\Users\USER\Desktop\dataset"  # 整理後のフォルダ
csv_file = os.path.join(output_dir, "dataset_labels.csv")  # CSVファイルのパス

# 分類のための関数
def determine_label(angle):
    """サーボ角度に基づいてラベルを決定"""
    if 0 <= angle <= 60:
        return "left"
    elif 61 <= angle <= 120:
        return "center"
    elif 121 <= angle <= 180:
        return "right"
    return None

# ディレクトリ作成
for split in ["train", "val", "test"]:
    for label in ["left", "right", "center"]:
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

# サンプルデータ分割
# 例: データセットの7:2:1分割
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
total_images = len(image_files)
train_split = int(total_images * 0.7)
val_split = int(total_images * 0.9)

# 結果を保存するリスト
dataset_rows = []

# イメージファイルの分類と移動
for i, filename in enumerate(image_files):
    # サーボ角度をファイル名から取得
    angle = int(filename.split("_angle_")[1].split(".jpg")[0])
    label = determine_label(angle)
    if label is None:
        print(f"ラベルを決定できませんでした: {filename}")
        continue

    # データセット用途を決定
    if i < train_split:
        split = "train"
    elif i < val_split:
        split = "val"
    else:
        split = "test"

    # ファイルの移動先を決定
    src_path = os.path.join(image_dir, filename)
    dest_path = os.path.join(output_dir, split, label, filename)

    # ファイルを移動
    shutil.move(src_path, dest_path)
    print(f"Moved: {src_path} -> {dest_path}")

    # CSVデータに追加
    dataset_rows.append({"filename": filename, "split": split, "label": label})

# CSVファイルを作成または更新
df = pd.DataFrame(dataset_rows)
df.to_csv(csv_file, index=False)
print(f"CSVファイルを作成しました: {csv_file}")

print("イメージの整理とCSVファイルの作成が完了しました！")
