import threading
import cv2
import datetime
import Jetson.GPIO as GPIO
import time
import keyboard
import subprocess

# カメラ映像処理の関数
def camera_processing():
    # カメラを初期化（カメラデバイスが複数ある場合は番号を変更）
    cap = cv2.VideoCapture(0)  # 0はデフォルトカメラを指定

    if not cap.isOpened():  # カメラが開けない場合のエラーハンドリング
        print("カメラが開けませんでした")
        return

    # カメラの解像度を設定（640x480ピクセル）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 動画保存用の設定
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVIDコーデックを使用
    out = None  # 録画用のVideoWriterオブジェクト（初期はNone）
    recording = False  # 録画状態を管理するフラグ

    print("リアルタイム映像が開始します。")
    print("録画を開始するには 'r' キーを押してください。")
    print("録画を停止するには再度 'r' キーを押してください。")
    print("終了するには 'q' キーを押してください。")

    while True:
        # カメラからフレームを取得
        ret, frame = cap.read()
        if not ret:  # 映像が取得できなかった場合のエラーハンドリング
            print("映像を取得できませんでした")
            break

        # フレームをウィンドウに表示
        cv2.imshow('Live Feed', frame)

        # 録画中の場合、現在のフレームを動画ファイルに書き込む
        if recording and out is not None:
            out.write(frame)

        # キーボード入力の処理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 'q'キーで終了
            print("終了します。")
            if recording:  # 録画中であれば録画を停止
                recording = False
                out.release()
                print("録画を停止しました")
            break
        elif key == ord('r'):  # 'r'キーで録画の開始/停止を切り替え
            if not recording:  # 録画が停止中の場合
                # タイムスタンプを使用してファイル名を作成
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = f"recording_{timestamp}.avi"
                out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))  # 新しい動画ファイルを作成
                recording = True
                print(f"録画を開始しました: {filename}")
            else:  # 録画中の場合
                recording = False
                out.release()  # 現在の動画ファイルを閉じる
                print("録画を停止しました")

    # カメラとリソースの解放
    cap.release()
    if out is not None:
        out.release()  # VideoWriterのリソースを解放
    cv2.destroyAllWindows()  # OpenCVのすべてのウィンドウを閉じる

# モーター制御の関数
def motor_control():
    # Sudo権限でのコマンド実行用パスワード（安全性のため、実際の運用では外部から読み込むべき）
    sudo_password = " "

    # サブプロセスでコマンドを実行するヘルパー関数
    def run_command(command):
        full_command = f"echo {sudo_password} | sudo -S {command}"  # sudoコマンドを実行
        subprocess.run(full_command, shell=True, check=True)

    # 必要なツール（busybox）がインストールされているか確認し、インストールする
    try:
        subprocess.run("busybox --help", shell=True, check=True)
    except subprocess.CalledProcessError:  # busyboxがない場合
        run_command("apt update && apt install -y busybox")  # busyboxをインストール

    # 特定のレジスタにデータを書き込むコマンド（モーター制御の準備）
    commands = [
        "busybox devmem 0x700031fc 32 0x45",
        "busybox devmem 0x6000d504 32 0x2",
        "busybox devmem 0x70003248 32 0x46",
        "busybox devmem 0x6000d100 32 0x00"
    ]
    for command in commands:
        run_command(command)

    # GPIOピンの設定
    servo_pin = 33  # サーボモーター制御用ピン
    dc_motor_pwm_pin = 32  # DCモーターの速度制御用PWMピン
    dc_motor_dir_pin1 = 29  # DCモーターの方向制御ピン1
    dc_motor_dir_pin2 = 31  # DCモーターの方向制御ピン2

    # GPIOモードの設定
    GPIO.setmode(GPIO.BOARD)  # ピン番号を基準にBOARDモードを使用
    GPIO.setup(servo_pin, GPIO.OUT)  # サーボモーターのピンを出力に設定
    GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)  # DCモーターPWMピンを出力に設定
    GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)  # DCモーター方向ピン1を出力に設定
    GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)  # DCモーター方向ピン2を出力に設定

    # PWM信号を生成
    servo = GPIO.PWM(servo_pin, 50)  # サーボモーター（50Hzで制御）
    dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # DCモーター（1000Hzで制御）
    servo.start(0)  # 初期デューティサイクルを0に設定
    dc_motor_pwm.start(0)  # 初期デューティサイクルを0に設定

    # サーボモーターの角度を設定する関数
    def set_servo_angle(angle):
        duty_cycle = 2 + (angle / 18)  # 角度をデューティサイクルに変換
        servo.ChangeDutyCycle(duty_cycle)
        time.sleep(0.1)  # モーターが動くのを待つ
        servo.ChangeDutyCycle(0)  # 信号を停止

    # DCモーターの速度と方向を設定する関数
    def set_dc_motor(speed, direction):
        if direction == "forward":  # 前進の場合
            GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        elif direction == "backward":  # 後退の場合
            GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
        dc_motor_pwm.ChangeDutyCycle(speed)  # デューティサイクルで速度を調整

    try:
        current_servo_angle = 90  # サーボモーターの初期角度を90度に設定
        set_servo_angle(current_servo_angle)  # 初期位置に移動

        print("W, A, S, D キーでモーターを操作します。終了するには 'q' を押してください。")
        while True:
            # キーボードの入力に応じてモーターを制御
            if keyboard.is_pressed('w'):  # 'W' キーで前進
                print("W キー押下: 前進")
                set_dc_motor(70, "forward")
            elif keyboard.is_pressed('s'):  # 'S' キーで後退
                print("S キー押下: 後退")
                set_dc_motor(70, "backward")
            elif keyboard.is_pressed('a'):  # 'A' キーで左回転
                print("A キー押下: サーボ左回転")
                current_servo_angle = max(0, current_servo_angle - 10)  # 角度を10度減少
                set_servo_angle(current_servo_angle)
            elif keyboard.is_pressed('d'):  # 'D' キーで右回転
                print("D キー押下: サーボ右回転")
                current_servo_angle = min(180, current_servo_angle + 10)  # 角度を10度増加
                set_servo_angle(current_servo_angle)
            elif keyboard.is_pressed('q'):  # 'Q' キーで終了
                print("終了します。")
                break
            else:
                # 入力がない場合、モーターを停止
                set_dc_motor(0, "forward")

            time.sleep(0.1)  # キーボード入力の判定を繰り返す間隔
    finally:
        # 終了時にすべてのリソースを解放
        servo.stop()  # サーボモーターのPWM信号を停止
        dc_motor_pwm.stop()  # DCモーターのPWM信号を停止
        GPIO.cleanup()  # GPIOピンを解放

# スレッドの定義と起動
camera_thread = threading.Thread(target=camera_processing)  # カメラ処理スレッド
motor_thread = threading.Thread(target=motor_control)  # モーター制御スレッド

camera_thread.start()  # カメラ処理を開始
motor_thread.start()  # モーター制御を開始

# スレッドが終了するまで待機
camera_thread.join()
motor_thread.join()
