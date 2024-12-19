import cv2
import os

# 入力フォルダと出力フォルダの設定
input_folder = r"C:\Users\USER\Desktop\RCcar_image"  # 入力フォルダ
output_folder = r"C:\Users\USER\Desktop\processed_images"  # 出力フォルダ

# 出力フォルダを作成（存在しない場合）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 画像処理ループ：フォルダ内のすべての画像を処理
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    print(f"Processing file: {filename}")

    # 画像ファイルのみ対象にする
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        img = cv2.imread(img_path)
        if img is not None:
            # カラー画像のままリサイズ
            resized_img = cv2.resize(img, (200, 66))

            # 出力フォルダに保存
            output_path = os.path.join(output_folder, f"processed_{filename}")
            cv2.imwrite(output_path, resized_img)
            print(f"Saved processed image to: {output_path}")
        else:
            print(f"Failed to load image: {filename}")
    else:
        print(f"Skipped non-image file: {filename}")

print("すべての画像処理が完了しました！")
