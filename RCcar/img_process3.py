import os
import shutil
import random
from PIL import Image

# データセットのフォルダパスを指定
dataset_path = r"C:\Users\USER\RCcar\dataset3"  # 必要に応じて変更
target_count = 2000  # 目標枚数
image_size = (66, 66)  # リサイズサイズ

# 画像フォーマット
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# すべてのサブフォルダを取得
folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

# 画像のカウントを調整
for folder in folders:
    folder_path = os.path.join(dataset_path, folder)
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    # 現在の画像数
    current_count = len(images)
    
    if current_count == 0:
        print(f"⚠ フォルダ {folder} には画像がありません。スキップします。")
        continue

    # 足りない場合はランダムにコピー（反転や加工なし）
    if current_count < target_count:
        print(f"📢 フォルダ {folder}: {current_count} 枚 → {target_count} 枚に増やします。")
        while len(images) < target_count:
            selected_image = random.choice(images)  # ランダムな画像を選択
            src = os.path.join(folder_path, selected_image)
            new_name = f"copy_{len(images)}_{selected_image}"
            dest = os.path.join(folder_path, new_name)
            shutil.copy(src, dest)  # そのままコピー
            images.append(new_name)
    
    # 多すぎる場合はランダムに削除
    elif current_count > target_count:
        print(f"📢 フォルダ {folder}: {current_count} 枚 → {target_count} 枚に削減します。")
        to_delete = random.sample(images, current_count - target_count)
        for image in to_delete:
            os.remove(os.path.join(folder_path, image))

    # 画像リサイズ処理（反転や加工なし）
    print(f"📢 フォルダ {folder}: すべての画像を {image_size} にリサイズします。")
    for image in os.listdir(folder_path):
        if image.lower().endswith(valid_extensions):
            img_path = os.path.join(folder_path, image)
            img = Image.open(img_path)
            img = img.resize(image_size, Image.LANCZOS)  # 修正：Image.ANTIALIAS → Image.LANCZOS
            img.save(img_path)

print("✅ 処理が完了しました！")

