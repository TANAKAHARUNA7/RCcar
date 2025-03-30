import os
import shutil
import random

# ルートフォルダのパス
root_folder = r"C:\Users\USER\Desktop\dataset"  # 実際のフォルダパスに置き換えてください

# フォルダリスト
folders = ["60-74", "75-89", "90", "91-104", "105-120"]

# 目標枚数
target_count = 1000

for folder_name in folders:
    folder_path = os.path.join(root_folder, folder_name)
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    current_count = len(images)
    
    if current_count < target_count:
        # 不足している場合：画像をランダムに複製
        print(f"{folder_name} is below target ({current_count}). Duplicating images...")
        while len(images) < target_count:
            image_to_copy = random.choice(images)
            new_image_name = f"copy_{len(images)}_{image_to_copy}"
            shutil.copy(os.path.join(folder_path, image_to_copy), os.path.join(folder_path, new_image_name))
            images.append(new_image_name)  # 複製した画像をリストに追加
    elif current_count > target_count:
        # 超過している場合：ランダムに画像を削除
        print(f"{folder_name} is above target ({current_count}). Deleting images...")
        images_to_remove = random.sample(images, current_count - target_count)
        for image in images_to_remove:
            os.remove(os.path.join(folder_path, image))
    else:
        print(f"{folder_name} is already at target ({current_count}). No changes needed.")

print("All folders have been balanced to 1000 images.")
