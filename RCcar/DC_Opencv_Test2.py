import threading  # スレッド処理を行うためのライブラリ
import cv2  # OpenCVライブラリを使用して映像処理を行う
import datetime  # タイムスタンプ作成用
import Jetson.GPIO as GPIO  # Jetson NanoのGPIOピン制御用ライブラリ
import time  # 遅延処理を行うためのライブラリ
import keyboard  # キーボード入力を検出するためのライブラリ

# カメラ映像処理の関数
def camera_processing():
    # カメラを初期化（デフォルトカメラを使用する場合は引数を0にする）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けませんでした")  # カメラが使用不可の場合にエラー表示
        return

    # カメラ解像度を設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 幅を640ピクセルに設定
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 高さを480ピクセルに設定

    print("リアルタイム映像が開始します。")
    print("写真を撮るには 'p' キーを押してください。")
    print("終了するには 'q' キーを押してください。")

    while True:
        # カメラから1フレーム取得
        ret, frame = cap.read()
        if not ret:
            print("映像を取得できませんでした")  # フレームが取得できない場合
            break

        # フレームを表示
        cv2.imshow('Live Feed', frame)

        # キーボード入力を検出
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 'q' キーで処理終了
            print("終了します。")
            break
        elif key == ord('p'):  # 'p' キーで写真を撮影
            # タイムスタンプを用いて一意のファイル名を生成
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"photo_{timestamp}.jpg"
            cv2.imwrite(filename, frame)  # フレームをJPEG画像として保存
            print(f"写真を保存しました: {filename}")

    # カメラとウィンドウを解放
    cap.release()
    cv2.destroyAllWindows()

# モータ制御の関数
def motor_control():
    # GPIOピン番号の定義（BOARDモード使用）
    servo_pin = 33  # サーボモーター制御用ピン番号
    dc_motor_pwm_pin = 32  # DCモーターのPWM制御用ピン番号
    dc_motor_dir_pin1 = 29  # DCモーターの方向制御ピン1
    dc_motor_dir_pin2 = 31  # DCモーターの方向制御ピン2

    # GPIOモードを設定
    GPIO.setmode(GPIO.BOARD)  # BOARDモード（物理的なピン番号）を使用
    GPIO.setup(servo_pin, GPIO.OUT)  # サーボモーター用ピンを出力モードに設定
    GPIO.setup(dc_motor_pwm_pin, GPIO.OUT)  # PWMピンを出力モードに設定
    GPIO.setup(dc_motor_dir_pin1, GPIO.OUT)  # 方向制御用ピン1を出力モードに設定
    GPIO.setup(dc_motor_dir_pin2, GPIO.OUT)  # 方向制御用ピン2を出力モードに設定

    # PWM信号の初期化
    servo = GPIO.PWM(servo_pin, 50)  # サーボモーター用PWM、50Hz（20ms周期）
    dc_motor_pwm = GPIO.PWM(dc_motor_pwm_pin, 1000)  # DCモーター用PWM、1000Hz
    servo.start(0)  # サーボモーターのPWMを0%デューティで開始（初期化）
    dc_motor_pwm.start(0)  # DCモーターのPWMを0%デューティで開始（初期化）

    # サーボモーターの角度を設定する関数
    def set_servo_angle(angle):
        """
        指定された角度にサーボモーターを回転させる関数。
        :param angle: サーボモーターの目標角度（0～180度）
        """
        # 角度をデューティ比（2～12%）に変換
        duty_cycle = 2 + (angle / 18)  # データシートに基づく変換式
        servo.ChangeDutyCycle(duty_cycle)  # PWMデューティ比を設定
        time.sleep(0.1)  # 短い遅延を加えてサーボモーターの動作を安定させる
        servo.ChangeDutyCycle(0)  # 停止信号を送信して角度を維持

    # DCモーターの速度と方向を設定する関数
    def set_dc_motor(speed, direction):
        """
        DCモーターの動作を制御する関数。
        :param speed: モーター速度（0～100%）
        :param direction: モーター回転方向（"forward" または "backward"）
        """
        if direction == "forward":
            # 前進方向: ピン1をHIGH、ピン2をLOWに設定
            GPIO.output(dc_motor_dir_pin1, GPIO.HIGH)
            GPIO.output(dc_motor_dir_pin2, GPIO.LOW)
        elif direction == "backward":
            # 後退方向: ピン1をLOW、ピン2をHIGHに設定
            GPIO.output(dc_motor_dir_pin1, GPIO.LOW)
            GPIO.output(dc_motor_dir_pin2, GPIO.HIGH)
        # PWMのデューティ比を設定して速度を制御
        dc_motor_pwm.ChangeDutyCycle(speed)

    try:
        # 初期状態: サーボモーターを90度（中央位置）に設定
        current_servo_angle = 90
        set_servo_angle(current_servo_angle)  # サーボモーターを中央に位置付け
        motor_running = False  # モーターが動作中かどうかを示すフラグ

        print("Wキーを押すとDCモーターが前進します。")
        print("Sキーを押すとモーターを停止します。")
        print("Aキーを押すとサーボモーターが左回転します。")
        print("Dキーを押すとサーボモーターが右回転します。")
        print("Qキーを押すと終了します。")

        while True:
            if keyboard.is_pressed('w') and not motor_running:
                print("Wキーが押されました: DCモーターを前進します。")
                set_dc_motor(70, "forward")  # 前進速度50%でモーターを動作
                motor_running = True  # モーターが動作中であることを設定
            elif keyboard.is_pressed('s'):
                print("Sキーが押されました: DCモーターを停止します。")
                set_dc_motor(0, "forward")  # モーターを停止（速度0%）
                motor_running = False  # モーターを停止状態に設定
            elif keyboard.is_pressed('a'):
                print("Aキーが押されました: サーボモーターを左回転します。")
                current_servo_angle = max(0, current_servo_angle - 5)  # 左回転（角度を減少）
                set_servo_angle(current_servo_angle)
            elif keyboard.is_pressed('d'):
                print("Dキーが押されました: サーボモーターを右回転します。")
                current_servo_angle = min(180, current_servo_angle + 5)  # 右回転（角度を増加）
                set_servo_angle(current_servo_angle)
            elif keyboard.is_pressed('q'):  # 'q' キーでループ終了
                print("Qキーが押されました: プログラムを終了します。")
                break

            time.sleep(0.1)  # 入力検出の間隔を短い遅延で設定
    finally:
        # モーターを安全に停止し、GPIOを解放
        print("モーターとGPIOをクリーンアップします。")
        servo.stop()  # サーボモーターのPWMを停止
        dc_motor_pwm.stop()  # DCモーターのPWMを停止
        GPIO.cleanup()  # GPIOリソースを解放してクリーンアップ

# カメラとモータ制御を並行実行するスレッドを作成
camera_thread = threading.Thread(target=camera_processing)
motor_thread = threading.Thread(target=motor_control)

# 並行処理を開始
camera_thread.start()
motor_thread.start()

# スレッドの終了を待機
camera_thread.join()
motor_thread.join()
