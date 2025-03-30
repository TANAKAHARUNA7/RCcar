import os
import shutil
import random

# メインフォルダ（copy2）のパスを指定
main_folder = r"C:\Users\USER\Desktop\copy2"  # 必要に応じて修正してください

# 目標枚数
target_count = 1000

# メインフォルダ内のサブフォルダを取得
folders = [os.path.join(main_folder, folder) for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]

for folder_path in folders:
    folder_name = os.path.basename(folder_path)  # フォルダ名（例: "60-74"）
    
    # フォルダ内の画像リストを取得
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    current_count = len(images)
    
    if current_count < target_count:
        # 不足している場合: ランダムに選択して複製
        print(f"{folder_name} is below target ({current_count}). Adding images...")
        while len(images) < target_count:
            image_to_copy = random.choice(images)
            # 新しい画像名を生成
            base_name, ext = os.path.splitext(image_to_copy)
            new_image_name = f"{base_name}_copy_{len(images)}{ext}"
            shutil.copy(os.path.join(folder_path, image_to_copy), os.path.join(folder_path, new_image_name))
            images.append(new_image_name)
    elif current_count > target_count:
        # 超過している場合: ランダムに削除
        print(f"{folder_name} is above target ({current_count}). Removing images...")
        images_to_remove = random.sample(images, current_count - target_count)
        for image in images_to_remove:
            os.remove(os.path.join(folder_path, image))
    else:
        print(f"{folder_name} is already at target ({current_count}). No changes needed.")

print("All folders have been balanced to 1000 images.")
