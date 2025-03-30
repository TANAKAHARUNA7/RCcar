import os
from PIL import Image

# 元フォルダパス
root_folder = r"C:\Users\USER\Desktop\copy2"  # 実際のフォルダパスに置き換え
folders = ["60-74", "75-89", "90", "91-104", "105-120"]

# 出力先フォルダ（上書き保存する場合は同じパスを指定）
output_root_folder = r"C:\Users\USER\Desktop\processed_images"

# 画像サイズ
resize_size = (64, 64)

# 処理の実行
for folder_name in folders:
    input_folder_path = os.path.join(root_folder, folder_name)
    output_folder_path = os.path.join(output_root_folder, folder_name)
    
    # 出力先フォルダがない場合は作成
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # 画像を1つずつ処理
    for image_name in os.listdir(input_folder_path):
        if image_name.endswith(('.jpg', '.png')):  # 対応する画像形式を指定
            input_image_path = os.path.join(input_folder_path, image_name)
            output_image_path = os.path.join(output_folder_path, image_name)
            
            # 画像を開く
            with Image.open(input_image_path) as img:
                # 元画像のサイズ取得
                width, height = img.size
                
                # 上部20%をカット
                crop_area = (0, int(height * 0.2), width, height)  # (left, upper, right, lower)
                cropped_img = img.crop(crop_area)
                
                # 64x64にリサイズ
                resized_img = cropped_img.resize(resize_size)
                
                # 画像を保存（上書きしたくない場合は別のフォルダを指定）
                resized_img.save(output_image_path)

print("Processing completed. All images are cropped and resized.")
