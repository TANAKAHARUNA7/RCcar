import torch
import Jetson.GPIO as GPIO
import cv2
import time
from torchvision import transforms
from PIL import Image
import locale
import sys

# エンコーディングの設定
locale.setlocale(locale.LC_ALL, 'C.UTF-8')

# モデル定義
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 16 * 16, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

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
num_classes = 5  # クラス数に合わせて設定
model = SimpleCNN(num_classes)
model.load_state_dict(torch.load("made_model.pth"))  # モデルパラメータをロード
model.eval()  # 推論モードに切り替え

# データ変換（入力画像の前処理）
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # トレーニング時と同じ正規化パラメータ
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

# 推論と制御
try:
    # カメラの初期化
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    while True:
        # カメラから画像をキャプチャ
        ret, frame = camera.read()
        if not ret:
            continue

        # 画像を前処理してモデルに入力
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

        # DCモーターを制御（前進）
        control_dc_motor(speed=50, direction="forward")
        time.sleep(0.2)

except KeyboardInterrupt:
    print("Program interrupted")

finally:
    print("Cleaning up GPIO...")
    camera.release()
    dc_pwm.stop()
    sv_pwm.stop()
    GPIO.cleanup()
