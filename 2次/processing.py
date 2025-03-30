import os
import cv2
import random
from glob import glob

# 원본 데이터셋 폴더 및 전처리 후 데이터셋 저장 폴더
BASE_DIR = r'C:\data\dataset'
OUTPUT_DIR = r'C:\data\processed_dataset'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 크기 조정 함수
def preprocess_image(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 열 수 없습니다: {image_path}")
        return None
    # 크기 조정
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)  # dsize를 target_size로 설정
    return resized_img

# 2. 데이터 균등화 함수 - 각 폴더당 고정된 수로 맞추기
def balance_data_to_fixed_count(folders, fixed_count=2000, target_size=(64, 64), output_format="jpg"):
    print(f"모든 폴더의 이미지를 {fixed_count}장으로 맞춥니다.")

    for folder in folders:
        image_files = glob(os.path.join(folder, "*.jpg"))
        folder_name = os.path.basename(folder)
        output_folder = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(output_folder, exist_ok=True)

        # 이미지가 fixed_count보다 많은 경우 랜덤하게 선택
        if len(image_files) > fixed_count:
            image_files = random.sample(image_files, fixed_count)

        # 기존 이미지 복사 및 전처리
        for image_path in image_files:
            img = preprocess_image(image_path, target_size=target_size)
            if img is not None:
                filename = os.path.basename(image_path).split('.')[0]
                cv2.imwrite(os.path.join(output_folder, f"{filename}.{output_format}"), img)

        # 이미지가 fixed_count보다 적은 경우 랜덤 복제
        while len(os.listdir(output_folder)) < fixed_count:
            img_path = random.choice(image_files)
            img = preprocess_image(img_path, target_size=target_size)
            if img is not None:
                new_filename = f"{random.randint(100000, 999999)}.{output_format}"
                cv2.imwrite(os.path.join(output_folder, new_filename), img)

    print("데이터 균형 맞추기 및 전처리 완료!")

# 3. 폴더별 이미지 개수 계산 함수
def count_images_in_folders(folder_path):
    folders = [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    image_counts = {os.path.basename(folder): len(glob(os.path.join(folder, "*.jpg"))) for folder in folders}
    return image_counts

# 4. 데이터 분포 출력 함수
def print_image_distribution(image_counts):
    print("폴더별 이미지 개수:")
    for folder, count in image_counts.items():
        print(f"{folder}: {count}")

# 5. 통합 실행
if __name__ == "__main__":
    # 전처리 전 데이터 분포 확인
    print("Checking dataset distribution before preprocessing")
    pre_processed_counts = count_images_in_folders(BASE_DIR)
    print_image_distribution(pre_processed_counts)

    # 데이터 전처리 수행
    print("Performing data preprocessing and balancing")
    folders = [os.path.join(BASE_DIR, d) for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    balance_data_to_fixed_count(folders, fixed_count=2000, target_size=(64, 64), output_format="jpg")

    # 전처리 후 데이터 분포 확인
    print("Checking dataset distribution after preprocessing")
    post_processed_counts = count_images_in_folders(OUTPUT_DIR)
    print_image_distribution(post_processed_counts)
