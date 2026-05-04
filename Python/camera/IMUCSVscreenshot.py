import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import threading
import csv
import os
import board
import busio
from adafruit_bno055 import BNO055_I2C

# Créé csv avec screenshots. pas oublier les imports

i2c = busio.I2C(board.SCL, board.SDA)
sensor = BNO055_I2C(i2c)
CONFIDENCE_THRESHOLD = 0.3

def read_imu():
    accel = sensor.acceleration
    gyro = sensor.gyro
    euler = sensor.euler
    linear = sensor.linear_acceleration
    gravity = sensor.gravity

    return {
        # Accelerometer
        "accel_x": accel[0] if accel else None,
        "accel_y": accel[1] if accel else None,
        "accel_z": accel[2] if accel else None,
        # Gyroscope
        "gyro_x": gyro[0] if gyro else None,
        "gyro_y": gyro[1] if gyro else None,
        "gyro_z": gyro[2] if gyro else None,
        # Orientation
        "yaw": euler[0] if euler else None,
        "roll": euler[1] if euler else None,
        "pitch": euler[2] if euler else None,
        # Linear acceleration
        "linear_x": linear[0] if linear else None,
        "linear_y": linear[1] if linear else None,
        "linear_z": linear[2] if linear else None,
        # Gravity vector
        "gravity_x": gravity[0] if gravity else None,
        "gravity_y": gravity[1] if gravity else None,
        "gravity_z": gravity[2] if gravity else None,
    }

SAVE_DIR = "captures"
os.makedirs(SAVE_DIR, exist_ok=True)

for file in os.listdir(SAVE_DIR):
    file_path = os.path.join(SAVE_DIR, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

# Delete old CSV
CSV_FILE = "motion_log.csv"
if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)

#csv
csv_rows = []
program_start_time = time.time()

interpreter = tflite.Interpreter(
    model_path="/home/alice/movenet_lightning.tflite",
    num_threads=4
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=25,
    detectShadows=False
)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

parts = [
    "head",
    "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

def simplify_keypoints(kp):

    nose = kp[0]

    ls = kp[5]
    rs = kp[6]
    lh = kp[11]
    rh = kp[12]

    left_elbow = kp[7]
    right_elbow = kp[8]
    left_wrist = kp[9]
    right_wrist = kp[10]
    left_knee = kp[13]
    right_knee = kp[14]
    left_ankle = kp[15]
    right_ankle = kp[16]
    head = nose

    neck = [
        (ls[0] + rs[0]) / 2,
        (ls[1] + rs[1]) / 2,
        min(ls[2], rs[2])
    ]
    return np.array([
        head,
        neck,
        ls, rs,
        left_elbow, right_elbow,
        left_wrist, right_wrist,
        lh, rh,
        left_knee, right_knee,
        left_ankle, right_ankle
    ])


latest_frame = None
latest_keypoints = None
lock = threading.Lock()

def pose_worker():
    global latest_frame
    global latest_keypoints

    while True:

        if latest_frame is None:
            time.sleep(0.005)
            continue
        with lock:
            frame = latest_frame.copy()

        img = cv2.resize(frame, (192, 192))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        kp = interpreter.get_tensor(output_details[0]['index'])[0][0]
        kp = simplify_keypoints(kp)
        with lock:
            latest_keypoints = kp


threading.Thread(target=pose_worker, daemon=True).start()

last_capture_time = 0
CAPTURE_COOLDOWN = 1.0

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    display = frame.copy()
    with lock:
        latest_frame = frame.copy()
        keypoints = latest_keypoints

    fgmask = fgbg.apply(frame)

    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
    fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_DILATE,kernel,iterations=2)
    _, fgmask = cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)

    moving_parts = []

    if keypoints is not None:
        h, w = frame.shape[:2]

        for i, kp in enumerate(keypoints):
            y, x, conf = kp

            if conf < CONFIDENCE_THRESHOLD:
                continue

            px = int(x * w)
            py = int(y * h)
            cv2.circle(display, (px, py), 4, (255, 255, 255), -1)

            if 0 <= px < w and 0 <= py < h:
                if fgmask[py, px] > 0:

                    moving_parts.append(parts[i])
                    cv2.circle(display,(px, py),7,(0, 0, 255),-1)
                    cv2.putText(display,parts[i],(px + 5, py),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 0, 255),1)
    if moving_parts:
        text = "Moving: " + ", ".join(set(moving_parts))
        cv2.putText(display,text,(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 255),2)

    #screenshot /IMU data
    current_time = time.time()

    if moving_parts and (current_time - last_capture_time > CAPTURE_COOLDOWN):

        elapsed_time = round(current_time - program_start_time, 3)
        filename = f"motion_{elapsed_time:.3f}.jpg"
        filepath = os.path.join(SAVE_DIR, filename)

        cv2.imwrite(filepath, display)

        imu_data = read_imu()

        row = {"time_since_start": elapsed_time,"image_file": filename,"moving_parts": ", ".join(set(moving_parts))}
        
        row.update(imu_data)
        csv_rows.append(row)
        print(f"Saved: {filename}")
        last_capture_time = current_time

    cv2.imshow("Simplified Real-Time Body Motion", display)
    cv2.imshow("Mask", fgmask)
    if cv2.waitKey(1) & 0xFF == 27:
        break

#cleanup
cap.release()
cv2.destroyAllWindows()

#csv
if len(csv_rows) > 0:
    fieldnames = list(csv_rows[0].keys())
    with open(CSV_FILE, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"CSV saved: {CSV_FILE}")
else:
    print("No motion events recorded.")
