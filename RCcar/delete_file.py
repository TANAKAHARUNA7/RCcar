import os
import pandas as pd

# ファイルパス
csv_file = r"C:\Users\USER\Desktop\dataset_labels.csv"
train_dir = r"C:\Users\USER\Desktop\dataset\train"

# CSVを読み込み
data = pd.read_csv(csv_file)

# 存在しないファイルを取り除く
valid_data = []
for _, row in data.iterrows():
    file_path = os.path.join(train_dir, row['filename'])
    if row['split'] == 'train' and not os.path.exists(file_path):
        print(f"ファイルが存在しません: {file_path}")
    else:
        valid_data.append(row)

# 新しいCSVとして保存
new_data = pd.DataFrame(valid_data)
new_data.to_csv(csv_file, index=False)
print("存在しないファイルを削除した新しいCSVを保存しました。")
