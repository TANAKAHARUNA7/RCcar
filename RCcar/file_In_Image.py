import threading
import cv2
import Jetson.GPIO as GPIO
import time
import keyboard
import os

# 定数: 撮影枚数を記録するファイルと写真保存ディレクトリ
PHOTO_COUNT_FILE = "photo_count.txt"  # 写真の枚数を記録するファイル
PHOTO_SAVE_DIR = "/home/haruna/Downloads/RCcar"  # 写真を保存するディレクトリのパス

# 関数: 撮影枚数をファイルから読み込む
def load_photo_count():
    """
    撮影した写真の枚数を記録したファイルから読み込む。
    ファイルが存在しない場合は初期値として0を返す。
    """
    if os.path.exists(PHOTO_COUNT_FILE):  # ファイルが存在するか確認
        with open(PHOTO_COUNT_FILE, "r") as file:
            return int(file.read().strip())  # ファイルの内容を整数として返す
    return 0  # ファイルがない場合は0を返す

# 関数: 撮影枚数をファイルに保存する
def save_photo_count(count):
    """
    撮影した写真の枚数をファイルに保存する。
    """
    with open(PHOTO_COUNT_FILE, "w") as file:
        file.write(str(count))  # 数値を文字列に変換して保存

# 関数: 写真保存用のディレクトリを作成
def ensure_photo_directory():
    """
    写真を保存するディレクトリが存在しない場合に作成する。
    """
    if not os.path.exists(PHOTO_SAVE_DIR):  # ディレクトリが存在するか確認
        os.makedirs(PHOTO_SAVE_DIR)  # ディレクトリを作成

# グローバル変数: 初期設定
current_servo_angle = 90  # サーボモーターの初期角度
photo_count = load_photo_count()  # 撮影済み写真の枚数をファイルから読み込む

# 関数: カメラ映像を処理する
def camera_processing():
    """
    カメラ映像をリアルタイムで表示し、写真撮影を可能にする処理。
    'p'キーで写真を撮影、'q'キーで終了する。
    """
    global photo_count, current_servo_angle

    # 写真保存ディレクトリを確認または作成
    ensure_photo_directory()

    # カメラデバイスの初期化
    cap = cv2.VideoCapture(0)  # デフォルトカメラ（0番デバイス）を使用
    if not cap.isOpened():  # カメラが正常に開けなかった場合
        print("カメラが開けませんでした。")
        return

    # カメラ解像度の設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 幅を640ピクセルに設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 高さを480ピクセルに設定

    print("カメラ映像が開始されました。")
    print("写真を撮りたい場合は 'p' キーを押してください。終了するには 'q' キーを押してください。")

    while True:
        ret, frame = cap.read()  # 映像フレームを取得
        if not ret:  # フレームが取得できなかった場合
            print("映像を取得できませんでした。")
            break

        # 映像を表示
        cv2.imshow('リアルタイム映像', frame)

        key = cv2.waitKey(1) & 0xFF  # キーボード入力を検知
        if key == ord('q'):  # 'q'キーでループを終了
            print("終了します。")
            break
        elif key == ord('p'):  # 'p'キーで写真撮影
            photo_name = os.path.join(PHOTO_SAVE_DIR, f"photo_{photo_count}_angle_{current_servo_angle}.jpg")
            cv2.imwrite(photo_name, frame)  # 画像を保存
            photo_count += 1  # 撮影枚数を更新
            save_photo_count(photo_count)  # ファイルに保存
            print(f"写真を保存しました: {photo_name}")

    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()

# 関数: モーターを制御する
def motor_control():
    """
    キーボード操作でサーボモーターとDCモーターを制御する。
    'W': 前進, 'S': 後退, 'A': 左旋回, 'D': 右旋回, 'Q': 終了
    """
    global current_servo_angle

    # GPIOピンの設定
    servo_pin = 33  # サーボモーター用ピン
    dc_motor_pwm_pin = 32  # DCモーターPWM用ピン
    dc_motor_dir_pin1 = 29  # DCモーター方向制御用ピン1
    dc_motor_dir_pin2 = 31  # DCモーター方向制御用ピン2

    GPIO.setmode(GPIO.BOARD)  # GPIOをボード番号で設定
    GPIO.setup(servo_pin, GPIO.OUT)
    GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)
    GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)
    GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)

    # PWMの初期化
    servo = GPIO.PWM(servo_pin, 50)  # サーボモーター（50Hz）
    dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # DCモーター（1000Hz）
    servo.start(0)  # サーボモーターを初期化
    dc_motor_pwm.start(0)  # DCモーターを初期化

    # 内部関数: サーボモーターの角度を設定
    def set_servo_angle(angle):
        """サーボモーターを指定した角度に動かす"""
        global current_servo_angle
        duty_cycle = 2 + (angle / 18)  # 角度をデューティサイクルに変換
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.1)  # 角度変更の安定化待ち
        servo.ChangeDutyCycle(0)
        current_servo_angle = angle

    # 内部関数: DCモーターの速度と方向を設定
    def set_dc_motor(speed, direction):
        """DCモーターの速度と方向を設定する"""
        if direction == "forward":
            GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        elif direction == "backward":
            GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
        dc_motor_pwm.ChangeDutyCycle(speed)

    try:
        set_servo_angle(current_servo_angle)  # 初期角度を設定

        print("操作方法: 'W'（前進）, 'S'（後退）, 'A'（左旋回）, 'D'（右旋回）, 'Q'（終了）")
        while True:
            if keyboard.is_pressed('w'):
                print("前進中...")
                set_dc_motor(50, "forward")
            elif keyboard.is_pressed('s'):
                print("後退中...")
                set_dc_motor(50, "backward")
            elif keyboard.is_pressed('a'):
                print("左旋回中...")
                new_angle = max(0, current_servo_angle - 15)
                set_servo_angle(new_angle)
            elif keyboard.is_pressed('d'):
                print("右旋回中...")
                new_angle = min(180, current_servo_angle + 15)
                set_servo_angle(new_angle)
            elif keyboard.is_pressed('q'):
                print("終了キーが押されました。")
                break
            else:
                set_dc_motor(0, "forward")  # 停止
            time.sleep(0.1)
    finally:
        servo.stop()
        dc_motor_pwm.stop()
        GPIO.cleanup()

# スレッドの設定と起動
camera_thread = threading.Thread(target=camera_processing)
motor_thread = threading.Thread(target=motor_control)

camera_thread.start()
motor_thread.start()

# スレッド終了待ち
camera_thread.join()
motor_thread.join()
