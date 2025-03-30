import os
from PIL import Image

# 入力フォルダと出力フォルダの指定
input_folder = r"C:\Users\USER\Desktop\リサイズ"  # 入力フォルダ
output_folder = r"C:\Users\USER\Desktop\リサイズすみ"  # 出力フォルダ

# 出力フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# フォルダ内の画像を処理
for filename in os.listdir(input_folder):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, filename)

    # 画像ファイルのみ処理
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        try:
            # 画像を開く
            with Image.open(input_path) as img:
                # 上20%をカット
                width, height = img.size
                crop_height = int(height * 0.2)  # 上20%の高さ
                cropped_img = img.crop((0, crop_height, width, height))  # (left, upper, right, lower)

                # 64x64にリサイズ
                resized_img = cropped_img.resize((64, 64), Image.Resampling.LANCZOS)

                # 出力フォルダに保存
                resized_img.save(output_path)

                print(f"Processed: {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
