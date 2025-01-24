import os
import shutil
import pandas as pd

# パス設定
image_dir = r"C:\Users\USER\Desktop\RCcar_img"
output_dir = r"C:\Users\USER\Desktop\dataset"
csv_file = r"C:\Users\USER\Desktop\dataset\dataset_labels.csv"

# CSVファイルが存在しない場合は作成
if not os.path.exists(csv_file):
    # 初期データフレーム（空の列を作成）
    df = pd.DataFrame(columns=["filename", "split", "label"])
    df.to_csv(csv_file, index=False)
    print(f"CSVファイルを作成しました: {csv_file}")
else:
    # 既存のCSVを読み込み
    df = pd.read_csv(csv_file)
    print(f"既存のCSVファイルを読み込みました: {csv_file}")

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

