import Jetson.GPIO as GPIO
import torch
import time
from model import SteeringModel  # CNNモデルをインポート
import cv2
import numpy as np

# --- GPIOピン設定 ---
PWM_DC = 32
IN1_DC = 29
IN2_DC = 31
PWM_SV = 33

GPIO.setmode(GPIO.BOARD)  # ボード番号でピン指定
GPIO.setup(PWM_DC, GPIO.OUT)
GPIO.setup(IN1_DC, GPIO.OUT)
GPIO.setup(IN2_DC, GPIO.OUT)
GPIO.setup(PWM_SV, GPIO.OUT)

# PWM設定
dc_motor_pwm = GPIO.PWM(PWM_DC, 100)  # 100HzでPWM信号
sv_motor_pwm = GPIO.PWM(PWM_SV, 50)  # サーボモーターは50Hz

dc_motor_pwm.start(0)  # 初期値は停止
sv_motor_pwm.start(0)

# --- 学習済みモデルのロード ---
model = SteeringModel()
model.load_state_dict(torch.load("steering_model.pth", map_location=torch.device("cpu")))
model.eval()

# --- 角度調整関数 ---
def adjust_servo_angle(angle):
    # 予測された角度に基づいて、サーボモーターのPWMデューティサイクルを設定
    if 60 <= angle <= 74:
        return 7.5  # 70度
    elif 76 <= angle <= 89:
        return 8.0  # 80度
    elif angle == 90:
        return 9.0  # 90度
    elif 91 <= angle <= 104:
        return 10.0  # 100度
    elif 105 <= angle <= 120:
        return 11.0  # 110度
    else:
        return 9.0  # デフォルトは90度

# --- DCモーター制御関数 ---
def control_dc_motor(speed):
    if speed > 0:  # 前進
        GPIO.output(IN1_DC, GPIO.HIGH)
        GPIO.output(IN2_DC, GPIO.LOW)
    elif speed < 0:  # 後退
        GPIO.output(IN1_DC, GPIO.LOW)
        GPIO.output(IN2_DC, GPIO.HIGH)
    else:  # 停止
        GPIO.output(IN1_DC, GPIO.LOW)
        GPIO.output(IN2_DC, GPIO.LOW)
    
    # PWM速度調整
    dc_motor_pwm.ChangeDutyCycle(abs(speed))

# --- カメラ起動 ---
cap = cv2.VideoCapture(0)  # カメラID

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 前処理：画像を64x64にリサイズ
        image = cv2.resize(frame, (64, 64))
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # 正規化

        # モデルによる推論
        with torch.no_grad():
            predicted_angle = model(image).item()

        print(f"Predicted Steering Angle: {predicted_angle}")

        # サーボモーターの角度を設定
        servo_angle = adjust_servo_angle(predicted_angle)
        sv_motor_pwm.ChangeDutyCycle(servo_angle)

        # DCモーターを制御（速度は一定と仮定、必要に応じて変更可能）
        control_dc_motor(50)  # 速度50%で前進

        # キー入力で終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program stopped by user.")

finally:
    # 終了処理
    cap.release()
    cv2.destroyAllWindows()
    dc_motor_pwm.stop()
    sv_motor_pwm.stop()
    GPIO.cleanup()
    print("GPIO cleaned up and program terminated.")
