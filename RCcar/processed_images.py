import os
from PIL import Image

# メインフォルダ（copy2）のパスを指定
main_folder = r"C:\Users\USER\Desktop\copy2"  # 必要に応じて修正してください

# 保存先フォルダ（変換後の画像を保存するフォルダ）
output_folder = os.path.join(main_folder, "processed_images")
os.makedirs(output_folder, exist_ok=True)

# 画像のターゲットサイズ
target_size = (64, 64)

# メインフォルダ内のサブフォルダを取得
folders = [os.path.join(main_folder, folder) for folder in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder))]

for folder_path in folders:
    folder_name = os.path.basename(folder_path)  # フォルダ名（例: "60-74"）
    output_subfolder = os.path.join(output_folder, folder_name)  # サブフォルダごとに保存
    os.makedirs(output_subfolder, exist_ok=True)
    
    # フォルダ内の画像を取得
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
    
    for image_name in images:
        image_path = os.path.join(folder_path, image_name)
        output_image_path = os.path.join(output_subfolder, image_name)
        
        try:
            # 画像を開く
            with Image.open(image_path) as img:
                # 上から20%をカット
                width, height = img.size
                top_cut = int(height * 0.2)
                cropped_img = img.crop((0, top_cut, width, height))  # (left, upper, right, lower)

                # 64x64にリサイズ
                resized_img = cropped_img.resize(target_size, Image.ANTIALIAS)

                # 保存
                resized_img.save(output_image_path)
            print(f"Processed: {image_path} -> {output_image_path}")
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

print("All images have been processed and saved to:", output_folder)
