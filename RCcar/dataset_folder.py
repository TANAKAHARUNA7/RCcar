import os
import shutil
import pandas as pd

# パス設定　経路の設定
image_dir = r"C:\Users\USER\Desktop\RCcar_image"
output_dir = r"C:\Users\USER\Desktop\dataset"
csv_file = r"C:\Users\USER\Desktop\dataset\dataset_labels.csv"

# CSV読み込み
df = pd.read_csv(csv_file)

# フォルダ作成
for split in ["train", "val", "test"]:
    for label in ["left", "right", "center"]:
        os.makedirs(os.path.join(output_dir, split, label), exist_ok=True)

# ファイル移動
for _, row in df.iterrows():
    src_path = os.path.join(image_dir, row["filename"])
    dest_path = os.path.join(output_dir, row["split"], row["label"], row["filename"])
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
        print(f"Moved: {src_path} -> {dest_path}")
    else:
        print(f"File not found: {src_path}")

print("データセットの整理が完了しました！")

