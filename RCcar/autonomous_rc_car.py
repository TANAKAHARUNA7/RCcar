import torch
import cv2
import numpy as np
import threading
import Jetson.GPIO as GPIO
import time

# GPIOピンの設定
PWM_DC = 32  # DCモータのPWMピン
IN1_DC = 29  # DCモータの制御ピン1
IN2_DC = 31  # DCモータの制御ピン2
PWM_SV = 33  # サーボモーターのPWMピン

GPIO.setmode(GPIO.BOARD)
GPIO.setup(PWM_DC, GPIO.OUT)
GPIO.setup(IN1_DC, GPIO.OUT)
GPIO.setup(IN2_DC, GPIO.OUT)
GPIO.setup(PWM_SV, GPIO.OUT)

# PWMの設定
pwm_dc = GPIO.PWM(PWM_DC, 1000)  # DCモーター用 PWM (1kHz)
pwm_sv = GPIO.PWM(PWM_SV, 50)    # サーボモーター用 PWM (50Hz)
pwm_dc.start(0)  # 初期状態は停止
pwm_sv.start(7.5)  # 初期状態は 90 度

# CNNモデルの定義（再ロードのため）
class LaneFollowerCNN(torch.nn.Module):
    def __init__(self):
        super(LaneFollowerCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = torch.nn.Linear(128 * 8 * 8, 512)
        self.fc2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x  # 予測した角度

# モデルのロード
model = LaneFollowerCNN()
model.load_state_dict(torch.load("/home/jetson/lane_follower_cnn.pth"))
model.eval()

# グローバル変数
frame = None
angle = 90  # デフォルトは直進
running = True  # スレッドの制御用フラグ

# カメラのキャプチャ開始
cap = cv2.VideoCapture(0)  # Jetson NanoのUSBカメラ or CSIカメラ

# SVモーターの角度マッピング関数
def adjust_servo_angle(predicted_angle):
    if 60 <= predicted_angle <= 74:
        return 70
    elif 76 <= predicted_angle <= 89:
        return 80
    elif predicted_angle == 90:
        return 90
    elif 91 <= predicted_angle <= 105:
        return 100
    elif 106 <= predicted_angle <= 120:
        return 110
    return 90  # 予測外の角度は直進（デフォルト）

# スレッドでモデルの推論を行う関数
def inference_thread():
    global frame, angle, running

    while running:
        if frame is not None:
            # 画像の前処理
            img = cv2.resize(frame, (66, 66))
            img = np.transpose(img, (2, 0, 1)) / 255.0  # チャンネルの順番を変えて正規化
            img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

            # モデルによる推論
            with torch.no_grad():
                output = model(img_tensor)
                angle = output.item()

            # サーボモーターの角度を調整
            servo_angle = adjust_servo_angle(angle)
            duty_cycle = (servo_angle / 18) + 2.5  # 角度 → PWMのデューティ比
            pwm_sv.ChangeDutyCycle(duty_cycle)

# 推論スレッドを開始
thread = threading.Thread(target=inference_thread)
thread.start()

# DCモーター制御関数
def drive_forward():
    GPIO.output(IN1_DC, GPIO.HIGH)
    GPIO.output(IN2_DC, GPIO.LOW)
    pwm_dc.ChangeDutyCycle(50)  # 50%の速度

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # モーター制御（ハンドル制御）
    if angle < 90:
        print(f"推定角度: {angle:.2f} 度 ⬅ 左にハンドルを切る")
    elif angle > 90:
        print(f"推定角度: {angle:.2f} 度 ➡ 右にハンドルを切る")
    else:
        print(f"推定角度: {angle:.2f} 度 ⬆ 直進")

    drive_forward()  # 常に前進

    # 画像を表示（デバッグ用）
    cv2.imshow('Camera View', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 終了処理
running = False
thread.join()
pwm_dc.stop()
pwm_sv.stop()
GPIO.cleanup()
cap.release()
cv2.destroyAllWindows()
