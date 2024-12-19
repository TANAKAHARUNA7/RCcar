import os
import pandas as pd

# ファイルパス設定
csv_file = r"C:\Users\USER\Desktop\dataset\dataset_labels.csv"
train_dir = r"C:\Users\USER\Desktop\dataset\train"

# CSVの読み込み
data = pd.read_csv(csv_file)

# 不足ファイルの特定
missing_files = []
for _, row in data.iterrows():
    if row['split'] == 'train':
        file_path = os.path.join(train_dir, row['filename'])
        if not os.path.exists(file_path):
            missing_files.append(row['filename'])

# 不足ファイルの出力
if missing_files:
    print("存在しないファイル:")
    for file in missing_files:
        print(file)
else:
    print("すべてのファイルが存在します！")
