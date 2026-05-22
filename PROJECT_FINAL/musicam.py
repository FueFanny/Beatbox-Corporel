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

from adafruit_bno055 import BNO055_I2C
from gpiozero import DistanceSensor


# INITIALIZATION

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


# IMU
i2c = busio.I2C(board.SCL, board.SDA)
imu = BNO055_I2C(i2c)

# ULTRASONIC
ultrasonic = DistanceSensor(
    echo=24,
    trigger=23,
    max_distance=2.0
)


# AUDIO
pygame.mixer.init(
    frequency=44100,
    size=-16,
    channels=2,
    buffer=512
)

sounds = {
    "kick_soft":pygame.mixer.Sound("sounds/soft/kick.wav"),
    "kick_medium":pygame.mixer.Sound("sounds/medium/kick.wav"),
    "kick_hard":pygame.mixer.Sound("sounds/hard/kick.wav"),
    "false":pygame.mixer.Sound("sounds/false/false.wav")
}

sounds["kick_soft"].set_volume(1.0)
sounds["kick_medium"].set_volume(1.0)
sounds["kick_hard"].set_volume(1.0)
sounds["false"].set_volume(0.18)


# AUDIO STATE
current_channel = None
current_sound_name = None
current_volume = 0.0

last_kick_change = 0
last_play_time = 0

filtered_distance = 0.40

peak_accel = 0
peak_accel_time = 0

right_tilt_start = None


# PARAMETERS
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


# CSV
csv_rows = []
program_start_time = time.time()
last_capture_time = 0
last_csv_log_time = 0


interpreter = tflite.Interpreter(
    model_path="/home/alice/movenet_lightning.tflite",
    num_threads=4
)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# CAMERA

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*'MJPG')
)

# lower resolution for performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

# BACKGROUND SUBTRACTION

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=25,
    detectShadows=False
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))

# BODY PARTS

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

# IMU READ

def read_imu():
    accel = imu.acceleration
    gyro = imu.gyro
    euler = imu.euler
    linear = imu.linear_acceleration
    gravity = imu.gravity

    return {
        "accel_x":accel[0] if accel else None,
        "accel_y":accel[1] if accel else None,
        "accel_z":accel[2] if accel else None,
        "gyro_x":gyro[0] if gyro else None,
        "gyro_y":gyro[1] if gyro else None,
        "gyro_z":gyro[2] if gyro else None,
        "yaw":euler[0] if euler else None,
        "roll":euler[1] if euler else None,
        "pitch":euler[2] if euler else None,
        "linear_x":linear[0] if linear else None,
        "linear_y":linear[1] if linear else None,
        "linear_z":linear[2] if linear else None,
        "gravity_x":gravity[0] if gravity else None,
        "gravity_y":gravity[1] if gravity else None,
        "gravity_z":gravity[2] if gravity else None
    }

# AUDIO HELPERS

def clamp(val, low, high):
    return max(low, min(high, val))


def distance_to_volume(distance):
    normalized = 1.0 - (
        (distance - MIN_DISTANCE) /
        (START_DISTANCE - MIN_DISTANCE)
    )

    normalized = clamp(normalized, 0.0, 1.0)
    boosted = normalized ** 0.35
    return clamp(boosted, 0.35, 1.0)


def get_kick_from_acceleration(accel_mag):
    if accel_mag >= HARD_ACCEL:
        return "kick_hard"

    elif accel_mag >= MEDIUM_ACCEL:
        return "kick_medium"

    else:
        return "kick_soft"


def stop_sound():
    global current_channel
    global current_sound_name
    global current_volume

    if current_channel:
        current_channel.stop()

    current_channel = None
    current_sound_name = None
    current_volume = 0.0


def play_loop(sound_name, volume):
    global current_channel
    global current_sound_name
    global current_volume
    global last_kick_change

    current_time = time.time()

    if sound_name not in sounds:
        return

    should_change = (
        current_sound_name != sound_name and
        current_time - last_kick_change > KICK_CHANGE_COOLDOWN
    )

    if current_channel is None:
        current_channel = sounds[sound_name].play(loops=-1)

        current_sound_name = sound_name
        last_kick_change = current_time

    elif should_change:
        stop_sound()

        current_channel = sounds[sound_name].play(loops=-1)
        current_sound_name = sound_name
        last_kick_change = current_time

    boosted_volume = min(volume * 2.2, 1.0)
    if current_channel:
        current_channel.set_volume(boosted_volume)
    current_volume = boosted_volume


# POSE SIMPLIFICATION

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

    return np.array([head,neck,ls,rs,left_elbow,right_elbow,left_wrist,right_wrist,lh,rh,left_knee,right_knee,left_ankle,right_ankle])

# THREAD SHARED DATA

latest_frame = None
latest_keypoints = None
lock = threading.Lock()

# POSE THREAD

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
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.uint8)

        interpreter.set_tensor(
            input_details[0]['index'],
            img
        )
        interpreter.invoke()
        kp = interpreter.get_tensor(output_details[0]['index'])[0][0]
        kp = simplify_keypoints(kp)

        with lock:
            latest_keypoints = kp

threading.Thread(target=pose_worker,daemon=True).start()

# MAIN LOOP

print("Programme lancé.")

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        display = frame.copy()

        with lock:
            latest_frame = frame.copy()
            keypoints = latest_keypoints

        # MOTION MASK

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
        fgmask = cv2.morphologyEx(fgmask,cv2.MORPH_DILATE,kernel,iterations=2)
        _, fgmask = cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)

        moving_parts = []

        # KEYPOINTS

        if keypoints is not None:
            h, w = frame.shape[:2]
            for i, kp in enumerate(keypoints):
                y, x, conf = kp
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                px = int(x * w)
                py = int(y * h)

                cv2.circle(
                    display,
                    (px, py),
                    4,
                    (255, 255, 255),
                    -1
                )

                if 0 <= px < w and 0 <= py < h:
                    if fgmask[py, px] > 0:
                        moving_parts.append(parts[i])

                        cv2.circle(
                            display,
                            (px, py),
                            7,
                            (0, 0, 255),
                            -1
                        )

                        cv2.putText(
                            display,
                            parts[i],
                            (px + 5, py),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1
                        )

        # MOTION TEXT

        if moving_parts:

            text = ("Moving: " + ", ".join(set(moving_parts)))

            cv2.putText(
                display,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

        current_time = time.time()

        # IMU

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

        raw_distance = ultrasonic.distance
        filtered_distance = (SMOOTHING * filtered_distance + (1 - SMOOTHING) * raw_distance)
        distance_cm = filtered_distance * 100

        if accel_mag > peak_accel:
            peak_accel = accel_mag
            peak_accel_time = current_time

        if current_time - peak_accel_time > MOTION_MEMORY_TIME:
            peak_accel *= 0.92

        effective_accel = max(accel_mag,peak_accel)

        tilt = "center"

        if roll is not None:
            if roll > TILT_THRESHOLD:
                tilt = "right"
            elif roll < -TILT_THRESHOLD:
                tilt = "left"

        hand_in_range = (MIN_DISTANCE <= filtered_distance <= START_DISTANCE)

        valid_pose = (tilt == "left" and hand_in_range)
        cheating_pose = (tilt == "right" and hand_in_range)

        if (valid_pose and current_time - last_play_time > MIN_PLAY_TIME):
            volume = distance_to_volume(filtered_distance)
            kick_type = get_kick_from_acceleration(effective_accel)

            play_loop(kick_type,volume)
            last_play_time = current_time

        else:
            stop_sound()

        if cheating_pose:
            if right_tilt_start is None:
                right_tilt_start = current_time

            held_time = (current_time - right_tilt_start)

            if held_time > FALSE_TRIGGER_TIME:
                sounds["false"].play()
                right_tilt_start = None
        else:
            right_tilt_start = None

        cv2.putText(
            display,
            f"Sound: {current_sound_name}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Volume: {current_volume:.2f}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )

        cv2.putText(
            display,
            f"Distance: {distance_cm:.1f} cm",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2
        )

        filename = None

        if (moving_parts and current_time - last_capture_time > CAPTURE_COOLDOWN):
            elapsed_time = round(current_time - program_start_time,3)

            filename = f"motion_{elapsed_time:.3f}.jpg"
            filepath = os.path.join(SAVE_DIR,filename)
            moving_points = []

            h, w = frame.shape[:2]

            for i, kp in enumerate(keypoints):
                y, x, conf = kp

                if conf < CONFIDENCE_THRESHOLD:
                    continue
                px = int(x * w)
                py = int(y * h)

                if parts[i] in moving_parts:
                    moving_points.append((px, py))

            if moving_points:
                xs = [p[0] for p in moving_points]
                ys = [p[1] for p in moving_points]

                x_min = min(xs)
                x_max = max(xs)

                y_min = min(ys)
                y_max = max(ys)

                padding = 60

                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)

                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)

                MIN_SIZE = 180

                box_w = x_max - x_min
                box_h = y_max - y_min

                if box_w < MIN_SIZE:
                    extra = (MIN_SIZE - box_w) // 2
                    x_min = max(0, x_min - extra)
                    x_max = min(w, x_max + extra)

                if box_h < MIN_SIZE:
                    extra = (MIN_SIZE - box_h) // 2
                    y_min = max(0, y_min - extra)
                    y_max = min(h, y_max + extra)

                cropped = display[
                    y_min:y_max,
                    x_min:x_max
                    ]

                if cropped.size > 0:
                    cv2.imwrite(filepath,cropped)
                    print(f"Saved: {filename}")

            last_capture_time = current_time

        if (current_time - last_csv_log_time > CSV_LOG_INTERVAL):

            row = {
                "time_since_start":
                    round(
                        current_time -
                        program_start_time,
                        3
                    ),
                "image_file":
                    filename,
                "moving_parts":
                    ", ".join(set(moving_parts)),
                "sound_played":
                    current_sound_name,
                "sound_volume":
                    round(current_volume, 3),
                "distance_cm":
                    round(distance_cm, 2)
            }
            row.update(imu_data)
            csv_rows.append(row)
            last_csv_log_time = current_time

        cv2.imshow(
            "Interactive Motion + Sound",
            display
        )
        cv2.imshow(
            "Motion Mask",
            fgmask
        )

        # ESC TO EXIT
        if cv2.waitKey(1) & 0xFF == 27:
            break

    except KeyboardInterrupt:
        break

    except Exception as e:
        print("Error:", e)
        time.sleep(0.05)

cap.release()
cv2.destroyAllWindows()
stop_sound()
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
else:
    print("\nNo data recorded.")
