import cv2
import os

# 入力フォルダと出力フォルダの設定
input_folder = r"C:\Users\USER\Desktop\RCcar_image"

output_folder = r"C:\Users\USER\Desktop\processed_images"

# 出力フォルダを作成（存在しない場合）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# ファイル処理を小分けにして実行
count = 0
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    print(f"\n--- Checking file: {filename} ---")
    print(f"Full Path: {img_path}")

    # 画像ファイルかどうか確認
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        print("This is a valid image file.")

        # 画像読み込み
        img = cv2.imread(img_path)
        if img is not None:
            print(f"Image loaded successfully! Shape: {img.shape}")

            # グレースケール変換とリサイズ
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_img = cv2.resize(gray_img, (200, 66))
            print("Image processed: Converted to grayscale and resized.")

            # 出力パスを確認
            output_path = os.path.join(output_folder, f"processed_{filename}")
            print(f"Saving to: {output_path}")

            # 画像保存確認
            success = cv2.imwrite(output_path, resized_img)
            if success:
                print(f"Saved successfully to: {output_path}")
            else:
                print(f"Failed to save image to: {output_path}")
            
            count += 1
            if count >= 5:  # 最初の5個だけ処理して確認
                print("\nProcessed 5 files, stopping.")
                break
        else:
            print("Failed to load image! Check the file format or file integrity.")
    else:
        print("Skipped: Not a valid image file.")

print("\n--- 画像処理が完了しました ---")



