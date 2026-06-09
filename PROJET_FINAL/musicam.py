import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import threading
import csv
import os
import math
import pygame
import board
import busio
import signal
import sys

from adafruit_bno055 import BNO055_I2C
from gpiozero import DistanceSensor


# =========================================================
# INITIALIZATION
# =========================================================

CONFIDENCE_THRESHOLD = 0.3

SAVE_DIR = "captures"
CSV_FILE = "motion_log.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

# clear old captures
for file in os.listdir(SAVE_DIR):

    file_path = os.path.join(SAVE_DIR, file)

    if os.path.isfile(file_path):
        os.remove(file_path)

# clear old csv
if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)


# =========================================================
# IMU
# =========================================================

i2c = busio.I2C(board.SCL, board.SDA)
imu = BNO055_I2C(i2c)


# =========================================================
# ULTRASONIC SENSORS
# =========================================================

left_sensor = DistanceSensor(
    echo=24,
    trigger=23,
    max_distance=2.0
)

right_sensor = DistanceSensor(
    echo=17,
    trigger=27,
    max_distance=2.0
)


# =========================================================
# AUDIO
# =========================================================

pygame.mixer.init(
    frequency=44100,
    size=-16,
    channels=2,
    buffer=512
)

sounds = {
    "kick_soft": pygame.mixer.Sound("sounds/soft/kick.wav"),
    "kick_medium": pygame.mixer.Sound("sounds/medium/kick.wav"),
    "kick_hard": pygame.mixer.Sound("sounds/hard/kick.wav"),
    "false": pygame.mixer.Sound("sounds/false/false.wav")
}

sounds["kick_soft"].set_volume(100.0)
sounds["kick_medium"].set_volume(100.0)
sounds["kick_hard"].set_volume(100.0)
sounds["false"].set_volume(0.18)


# =========================================================
# AUDIO STATE
# =========================================================

current_channel = None
current_sound_name = None

last_play_time = 0

filtered_left_distance = 0.40
filtered_right_distance = 0.40

peak_accel = 0
peak_accel_time = 0

left_was_in_range = False
right_was_in_range = False

last_false_time = 0
FALSE_COOLDOWN = 0.3
# =========================================================
# PARAMETERS
# =========================================================

TILT_THRESHOLD = 15

START_DISTANCE = 0.40
MIN_DISTANCE = 0.03

SMOOTHING = 0.75

KICK_CHANGE_COOLDOWN = 0.65

FALSE_TRIGGER_TIME = 0.55

MEDIUM_ACCEL = 1.3
HARD_ACCEL = 2.8

MOTION_MEMORY_TIME = 0.45

MIN_PLAY_TIME = 0.4

CAPTURE_COOLDOWN = 0.5

CSV_LOG_INTERVAL = 0.1


# =========================================================
# CSV
# =========================================================

csv_rows = []

program_start_time = time.time()

last_capture_time = 0
last_csv_log_time = 0


# =========================================================
# TFLITE
# =========================================================

interpreter = tflite.Interpreter(
    model_path="/home/alice/movenet_lightning.tflite",
    num_threads=4
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*'MJPG')
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)


# =========================================================
# BODY PARTS
# =========================================================

parts = [
    "head",
    "neck",

    "left_shoulder",
    "right_shoulder",

    "left_elbow",
    "right_elbow",

    "left_wrist",
    "right_wrist",

    "left_hip",
    "right_hip",

    "left_knee",
    "right_knee",

    "left_ankle",
    "right_ankle"
]

connections = [
    (0, 1),

    (1, 2),
    (1, 3),

    (2, 4),
    (4, 6),

    (3, 5),
    (5, 7),

    (2, 8),
    (3, 9),

    (8, 9),

    (8, 10),
    (10, 12),

    (9, 11),
    (11, 13)
]


# =========================================================
# IMU READ
# =========================================================

def read_imu():

    accel = imu.acceleration
    gyro = imu.gyro
    euler = imu.euler
    linear = imu.linear_acceleration
    gravity = imu.gravity

    return {
        "accel_x": accel[0] if accel else None,
        "accel_y": accel[1] if accel else None,
        "accel_z": accel[2] if accel else None,

        "gyro_x": gyro[0] if gyro else None,
        "gyro_y": gyro[1] if gyro else None,
        "gyro_z": gyro[2] if gyro else None,

        "yaw": euler[0] if euler else None,
        "roll": euler[1] if euler else None,
        "pitch": euler[2] if euler else None,

        "linear_x": linear[0] if linear else None,
        "linear_y": linear[1] if linear else None,
        "linear_z": linear[2] if linear else None,

        "gravity_x": gravity[0] if gravity else None,
        "gravity_y": gravity[1] if gravity else None,
        "gravity_z": gravity[2] if gravity else None
    }


# =========================================================
# AUDIO HELPERS
# =========================================================

def clamp(val, low, high):
    return max(low, min(high, val))


def get_kick_from_acceleration(accel_mag):

    if accel_mag >= HARD_ACCEL:
        return "kick_hard"

    elif accel_mag >= MEDIUM_ACCEL:
        return "kick_medium"

    else:
        return "kick_soft"

def play_one_shot(sound_name):
    global current_sound_name
    if sound_name not in sounds:
        return
    sounds[sound_name].play()
    current_sound_name = sound_name

# =========================================================
# POSE SIMPLIFICATION
# =========================================================

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

        ls,
        rs,

        left_elbow,
        right_elbow,

        left_wrist,
        right_wrist,

        lh,
        rh,

        left_knee,
        right_knee,

        left_ankle,
        right_ankle
    ])


# =========================================================
# THREAD DATA
# =========================================================

latest_frame = None
latest_keypoints = None

lock = threading.Lock()

running = True


# =========================================================
# CLEAN EXIT
# =========================================================

def clean_exit(signum=None, frame=None):
    global running
    running = False
    print("\nStopping...")
    cap.release()
    cv2.destroyAllWindows()

    pygame.quit()

    if len(csv_rows) > 0:

        fieldnames = list(csv_rows[0].keys())

        with open(
            CSV_FILE,
            mode="w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\nCSV saved: {CSV_FILE}")

    sys.exit(0)


signal.signal(signal.SIGTERM, clean_exit)
signal.signal(signal.SIGINT, clean_exit)


# =========================================================
# POSE THREAD
# =========================================================

def pose_worker():

    global latest_frame
    global latest_keypoints

    print("Pose thread started")

    while running:

        try:

            if latest_frame is None:
                time.sleep(0.005)
                continue

            with lock:
                frame = latest_frame.copy()

            img = cv2.resize(frame, (192, 192))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = img.astype(np.float32)
            img = np.expand_dims(img, axis=0)

            interpreter.set_tensor(
                input_details[0]['index'],
                img
            )

            interpreter.invoke()

            kp = interpreter.get_tensor(
                output_details[0]['index']
            )[0][0]

            kp = simplify_keypoints(kp)

            with lock:
                latest_keypoints = kp

            print("Pose OK")

        except Exception as e:
            print("POSE THREAD ERROR:", e)
            time.sleep(0.1)


threading.Thread(
    target=pose_worker,
    daemon=True
).start()


# =========================================================
# MAIN LOOP
# =========================================================

print("Programme lancé.")

while running:

    try:

        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        display = frame.copy()

        with lock:
            latest_frame = frame.copy()
            keypoints = latest_keypoints

        moving_parts = []

        current_time = time.time()

        # =====================================================
        # KEYPOINTS
        # =====================================================

        if keypoints is not None:
            h, w = frame.shape[:2]

            # joints
            for i, kp in enumerate(keypoints):

                y, x, conf = kp
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                moving_parts.append(parts[i])
                px = int(x * w)
                py = int(y * h)
                cv2.circle(display,(px, py),6,(255, 255, 255),-1)

                cv2.putText(display,parts[i],(px + 5, py),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0, 255, 0),1)

        # =====================================================
        # IMU
        # =====================================================

        imu_data = read_imu()

        roll = imu_data["roll"]

        accel_x = imu_data["linear_x"] or 0
        accel_y = imu_data["linear_y"] or 0
        accel_z = imu_data["linear_z"] or 0

        accel_mag = math.sqrt(
            accel_x**2 +
            accel_y**2 +
            accel_z**2
        )

        # =====================================================
        # ULTRASONIC
        # =====================================================
        raw_left_distance = left_sensor.distance
        raw_right_distance = right_sensor.distance

        def valid_distance(d):
            return d is not None and 0.01 < d < 2.0

        if not valid_distance(raw_left_distance):
            raw_left_distance = filtered_left_distance

        if not valid_distance(raw_right_distance):
            raw_right_distance = filtered_right_distance

        filtered_left_distance = (
            SMOOTHING * filtered_left_distance +
            (1 - SMOOTHING) * raw_left_distance
        )

        filtered_right_distance = (
            SMOOTHING * filtered_right_distance +
            (1 - SMOOTHING) * raw_right_distance
        )

        left_distance_cm = filtered_left_distance * 100
        right_distance_cm = filtered_right_distance * 100

        # =====================================================
        # ACCEL MEMORY
        # =====================================================

        if accel_mag > peak_accel:

            peak_accel = accel_mag
            peak_accel_time = current_time

        if (
            current_time - peak_accel_time >
            MOTION_MEMORY_TIME
        ):

            peak_accel *= 0.92

        effective_accel = max(
            accel_mag,
            peak_accel
        )

        # =====================================================
        # TILT
        # =====================================================

        tilt = "center"

        if roll is not None:

            if roll > TILT_THRESHOLD:
                tilt = "right"

            elif roll < -TILT_THRESHOLD:
                tilt = "left"

        # =====================================================
        # SENSOR STATES
        # =====================================================

        left_in_range = (
            MIN_DISTANCE <= filtered_left_distance <= START_DISTANCE
        )

        right_in_range = (
            MIN_DISTANCE <= filtered_right_distance <= START_DISTANCE
        )

        correct_trigger = False
        wrong_trigger = False

        if tilt == "left":
            correct_trigger = left_in_range and not left_was_in_range
            wrong_trigger = right_in_range and not right_was_in_range

        elif tilt == "right":
            correct_trigger = right_in_range and not right_was_in_range
            wrong_trigger = left_in_range and not left_was_in_range

        left_was_in_range = left_in_range
        right_was_in_range = right_in_range

        # =====================================================
        # AUDIO
        # =====================================================

        if correct_trigger and (current_time - last_play_time > MIN_PLAY_TIME):
            kick_type = get_kick_from_acceleration(effective_accel)
            play_one_shot(kick_type)
            last_play_time = current_time

        # =====================================================
        # FALSE SOUND
        # =====================================================

        if wrong_trigger and (current_time - last_false_time > FALSE_COOLDOWN):
            sounds["false"].play()
            last_false_time = current_time

        # =====================================================
        # UI
        # =====================================================

        cv2.putText(
            display,
            f"Tilt: {tilt}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display,
            f"Left: {left_distance_cm:.1f} cm",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

        cv2.putText(
            display,
            f"Right: {right_distance_cm:.1f} cm",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            2
        )

        cv2.putText(
            display,
            f"Sound: {current_sound_name if current_sound_name else 'none'}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        # =====================================================
        # SCREENSHOT
        # =====================================================

        filename = None

        if (
            keypoints is not None and
            current_time - last_capture_time > CAPTURE_COOLDOWN
        ):

            elapsed_time = round(
                current_time - program_start_time,
                3
            )

            filename = (
                f"motion_{elapsed_time:.3f}.jpg"
            )

            filepath = os.path.join(
                SAVE_DIR,
                filename
            )

            cv2.imwrite(
                filepath,
                display
            )

            print(f"Saved: {filename}")

            last_capture_time = current_time

        # =====================================================
        # CSV
        # =====================================================

        if (
            current_time - last_csv_log_time >
            CSV_LOG_INTERVAL
        ):

            row = {
                "time_since_start": round(
                    current_time -
                    program_start_time,
                    3
                ),

                "image_file": filename,

                "sound_played":
                    current_sound_name,

                "left_distance_cm":
                    round(left_distance_cm, 2),

                "right_distance_cm":
                    round(right_distance_cm, 2),

                "tilt":
                    tilt,

                "correct_trigger":
                    correct_trigger,

                "wrong_trigger":
                    wrong_trigger
            }

            row.update(imu_data)

            csv_rows.append(row)

            last_csv_log_time = current_time

        # =====================================================
        # WINDOW
        # =====================================================

        cv2.imshow(
            "Interactive Motion + Sound",
            display
        )

        # =====================================================
        # ESC
        # =====================================================

        if cv2.waitKey(1) & 0xFF == 27:
            clean_exit()

    except KeyboardInterrupt:
        clean_exit()

    except Exception as e:

        print("Error:", e)

        time.sleep(0.05)

clean_exit()
