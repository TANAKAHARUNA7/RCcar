import threading
import torch
import Jetson.GPIO as GPIO
import cv2
import time
from torchvision import transforms
from PIL import Image

# GPIOピンの設定
DC_PWM_PIN = 32
DC_IN1_PIN = 29
DC_IN2_PIN = 31
SV_PWM_PIN = 33

# GPIOモードの設定
GPIO.setmode(GPIO.BOARD)
GPIO.setup(DC_PWM_PIN, GPIO.OUT)
GPIO.setup(DC_IN1_PIN, GPIO.OUT)
GPIO.setup(DC_IN2_PIN, GPIO.OUT)
GPIO.setup(SV_PWM_PIN, GPIO.OUT)

# PWMの初期化
dc_pwm = GPIO.PWM(DC_PWM_PIN, 100)  # DCモーター用PWM (周波数: 100Hz)
sv_pwm = GPIO.PWM(SV_PWM_PIN, 50)  # サーボモーター用PWM (周波数: 50Hz)
dc_pwm.start(0)
sv_pwm.start(7.5)  # サーボを90度に初期化

# 学習済みモデルのロード
model = torch.load("model.pth")
model.eval()

# データ変換（入力画像の前処理）
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# サーボモーターの角度をPWMデューティ比に変換
def angle_to_duty_cycle(angle):
    return 2.5 + (angle / 18.0)

# サーボモーターの制御
def control_servo(angle):
    duty_cycle = angle_to_duty_cycle(angle)
    sv_pwm.ChangeDutyCycle(duty_cycle)

# DCモーターの制御
def control_dc_motor(speed, direction):
    if direction == "forward":
        GPIO.output(DC_IN1_PIN, GPIO.HIGH)
        GPIO.output(DC_IN2_PIN, GPIO.LOW)
    elif direction == "backward":
        GPIO.output(DC_IN1_PIN, GPIO.LOW)
        GPIO.output(DC_IN2_PIN, GPIO.HIGH)
    elif direction == "stop":
        GPIO.output(DC_IN1_PIN, GPIO.LOW)
        GPIO.output(DC_IN2_PIN, GPIO.LOW)
    dc_pwm.ChangeDutyCycle(speed)

# 推論スレッド（サーボモーター制御）
def inference_thread(camera, stop_event):
    while not stop_event.is_set():
        ret, frame = camera.read()
        if not ret:
            continue

        # 画像を保存して推論
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)
            label = predicted.item()

            # ラベルに基づいてサーボモーターの角度を設定
            if label == 0:  # 105-120
                angle = 110
            elif label == 1:  # 60-74
                angle = 70
            elif label == 2:  # 75-89
                angle = 80
            elif label == 3:  # 90
                angle = 90
            elif label == 4:  # 91-104
                angle = 100
            else:
                angle = 90

            print(f"Predicted Angle: {angle}")
            control_servo(angle)

        time.sleep(0.2)

# DCモーター制御スレッド
def motor_thread(stop_event):
    while not stop_event.is_set():
        # DCモーターを動かす（前進）
        control_dc_motor(speed=50, direction="forward")
        time.sleep(0.1)  # 動作間隔調整

# メイン処理
try:
    # カメラの初期化
    camera = cv2.VideoCapture(0)  # カメラデバイスIDは適切に設定
    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    # スレッドの開始
    stop_event = threading.Event()
    inference = threading.Thread(target=inference_thread, args=(camera, stop_event))
    motor = threading.Thread(target=motor_thread, args=(stop_event,))

    inference.start()
    motor.start()

    # メインスレッドで停止待機
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopped by user")

finally:
    # スレッド停止
    stop_event.set()
    inference.join()
    motor.join()

    # リソースの解放
    camera.release()
    dc_pwm.stop()
    sv_pwm.stop()
    GPIO.cleanup()
