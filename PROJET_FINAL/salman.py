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
import pigpio   # ✅ added

from adafruit_bno055 import BNO055_I2C


# =========================================================
# INITIALIZATION
# =========================================================

CONFIDENCE_THRESHOLD = 0.3

SAVE_DIR = "captures"
CSV_FILE = "motion_log.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

for file in os.listdir(SAVE_DIR):
    file_path = os.path.join(SAVE_DIR, file)
    if os.path.isfile(file_path):
        os.remove(file_path)

if os.path.exists(CSV_FILE):
    os.remove(CSV_FILE)


# =========================================================
# IMU
# =========================================================

i2c = busio.I2C(board.SCL, board.SDA)
imu = BNO055_I2C(i2c)


# =========================================================
# ULTRASONIC (pigpio)
# =========================================================

TRIG_LEFT = 23
ECHO_LEFT = 24

TRIG_RIGHT = 27
ECHO_RIGHT = 17

pi = pigpio.pi()

pi.set_mode(TRIG_LEFT, pigpio.OUTPUT)
pi.set_mode(ECHO_LEFT, pigpio.INPUT)

pi.set_mode(TRIG_RIGHT, pigpio.OUTPUT)
pi.set_mode(ECHO_RIGHT, pigpio.INPUT)

pi.write(TRIG_LEFT, 0)
pi.write(TRIG_RIGHT, 0)

time.sleep(0.1)


def read_distance(trig, echo):
    pi.gpio_trigger(trig, 10)

    start = time.time()
    timeout = start + 0.02

    while pi.read(echo) == 0:
        if time.time() > timeout:
            return None
        start = time.time()

    while pi.read(echo) == 1:
        if time.time() > timeout:
            return None
        end = time.time()

    duration = end - start
    distance = (duration * 343) / 2
    return distance


# =========================================================
# AUDIO
# =========================================================

pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

sounds = {
    "kick_soft": pygame.mixer.Sound("sounds/soft/kick.wav"),
    "kick_medium": pygame.mixer.Sound("sounds/medium/kick.wav"),
    "kick_hard": pygame.mixer.Sound("sounds/hard/kick.wav"),
    "false": pygame.mixer.Sound("sounds/false/false.wav")
}

sounds["kick_soft"].set_volume(1.0)
sounds["kick_medium"].set_volume(1.0)
sounds["kick_hard"].set_volume(1.0)
sounds["false"].set_volume(0.18)


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
MEDIUM_ACCEL = 1.3
HARD_ACCEL = 2.8
MOTION_MEMORY_TIME = 0.45
MIN_PLAY_TIME = 0.4


# =========================================================
# CSV
# =========================================================

csv_rows = []
program_start_time = time.time()


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

cap = cv2.VideoCapture(0)


# =========================================================
# IMU READ
# =========================================================

def read_imu():
    accel = imu.linear_acceleration
    euler = imu.euler

    return {
        "linear_x": accel[0] if accel else 0,
        "linear_y": accel[1] if accel else 0,
        "linear_z": accel[2] if accel else 0,
        "roll": euler[1] if euler else 0
    }


# =========================================================
# AUDIO
# =========================================================

def get_kick_from_acceleration(accel):
    if accel >= HARD_ACCEL:
        return "kick_hard"
    elif accel >= MEDIUM_ACCEL:
        return "kick_medium"
    return "kick_soft"


def play_one_shot(name):
    global current_sound_name
    sounds[name].play()
    current_sound_name = name


# =========================================================
# CLEAN EXIT
# =========================================================

def clean_exit(signum=None, frame=None):
    global running
    running = False

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
    pi.stop()  # important

    if len(csv_rows) > 0:
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)

    sys.exit(0)


signal.signal(signal.SIGINT, clean_exit)


# =========================================================
# MAIN LOOP
# =========================================================

running = True

while running:
    ret, frame = cap.read()
    if not ret:
        continue

    current_time = time.time()

    # ================= ULTRASONIC =================
    raw_left_distance = read_distance(TRIG_LEFT, ECHO_LEFT)
    time.sleep(0.05)

    raw_right_distance = read_distance(TRIG_RIGHT, ECHO_RIGHT)
    time.sleep(0.05)

    print("L:", raw_left_distance, "R:", raw_right_distance)

    # ================= FILTER =================
    def valid(d):
        return d is not None and 0.01 < d < 2.0

    if valid(raw_left_distance):
        filtered_left_distance = SMOOTHING * filtered_left_distance + (1 - SMOOTHING) * raw_left_distance

    if valid(raw_right_distance):
        filtered_right_distance = SMOOTHING * filtered_right_distance + (1 - SMOOTHING) * raw_right_distance

    left_cm = filtered_left_distance * 100
    right_cm = filtered_right_distance * 100

    # ================= IMU =================
    imu_data = read_imu()

    accel = math.sqrt(
        imu_data["linear_x"]**2 +
        imu_data["linear_y"]**2 +
        imu_data["linear_z"]**2
    )

    roll = imu_data["roll"]

    tilt = "center"
    if roll > TILT_THRESHOLD:
        tilt = "right"
    elif roll < -TILT_THRESHOLD:
        tilt = "left"

    # ================= LOGIC =================
    left_in_range = MIN_DISTANCE <= filtered_left_distance <= START_DISTANCE
    right_in_range = MIN_DISTANCE <= filtered_right_distance <= START_DISTANCE

    correct = False

    if tilt == "left" and left_in_range:
        correct = True
    elif tilt == "right" and right_in_range:
        correct = True

    if correct and (current_time - last_play_time > MIN_PLAY_TIME):
        sound = get_kick_from_acceleration(accel)
        play_one_shot(sound)
        last_play_time = current_time

    # ================= DISPLAY =================
    cv2.putText(frame, f"L: {left_cm:.1f} cm", (10, 30), 0, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"R: {right_cm:.1f} cm", (10, 60), 0, 0.7, (255, 255, 255), 2)

    cv2.imshow("System", frame)

    if cv2.waitKey(1) == 27:
        clean_exit()

  #Before running
  #sudo systemctl start pigpiod
