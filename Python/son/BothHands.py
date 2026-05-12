import time
import math
import board
import busio
import pygame

from adafruit_bno055 import BNO055_I2C
from gpiozero import DistanceSensor

i2c = busio.I2C(board.SCL, board.SDA)
imu = BNO055_I2C(i2c)

left_sensor = DistanceSensor(
    echo=24,
    trigger=23,
    max_distance=2.0
)

right_sensor = DistanceSensor(
    echo=5,
    trigger=6,
    max_distance=2.0
)

pygame.mixer.init()
sounds = {
    "kick_soft": pygame.mixer.Sound("sounds/soft/kick.wav"),
    "kick_medium": pygame.mixer.Sound("sounds/medium/kick.wav"),
    "kick_hard": pygame.mixer.Sound("sounds/hard/kick.wav"),
    "false": pygame.mixer.Sound("sounds/false/false.wav")
}


TILT_THRESHOLD = 15

GROUND_THRESHOLD = 0.55
MIN_DISTANCE = 0.15

NOTE_COOLDOWN = 0.25
FALSE_TRIGGER_TIME = 0.45

MEDIUM_ACCEL = 2.0
HARD_ACCEL = 4.0

last_note_time = 0
wrong_pose_start = None

def clamp(val, low, high):
    return max(low, min(high, val))

def distance_to_volume(distance):
    normalized = 1.0 - ((distance - MIN_DISTANCE) / (GROUND_THRESHOLD - MIN_DISTANCE))
    normalized = clamp(normalized, 0.2, 1.0)
    return normalized

def get_kick_from_acceleration(accel_mag):
    if accel_mag > HARD_ACCEL:
        return "kick_hard"
    elif accel_mag > MEDIUM_ACCEL:
        return "kick_medium"
    else:
        return "kick_soft"

def play(name, volume=1.0):
    if name in sounds:
        sounds[name].set_volume(volume)
        sounds[name].play()


expected_sensor_for_tilt = {
    "left": "left",
    "right": "right"
}

print("Programme commence.")

while True:

    try:
        euler = imu.euler
        acceleration = imu.linear_acceleration
        if euler is None or acceleration is None:
            continue
        yaw, roll, pitch = euler

        tilt = "center"
        if roll > TILT_THRESHOLD:
            tilt = "right"
        elif roll < -TILT_THRESHOLD:
            tilt = "left"

        left_distance = left_sensor.distance
        right_distance = right_sensor.distance

        left_in_range = (MIN_DISTANCE <= left_distance <= GROUND_THRESHOLD)
        right_in_range = (MIN_DISTANCE <= right_distance <= GROUND_THRESHOLD)

        ax, ay, az = acceleration

        accel_mag = math.sqrt(ax * ax + ay * ay + az * az)

        print(
            f"Tilt: {tilt} | "
            f"Left: {left_distance*100:.1f} cm | "
            f"Right: {right_distance*100:.1f} cm | "
            f"Accel: {accel_mag:.2f}"
        )

        current_time = time.time()
        expected_sensor = expected_sensor_for_tilt.get(tilt)

        correct_trigger = False
        wrong_trigger = False
        active_distance = None

        if expected_sensor == "left":

            if left_in_range:
                correct_trigger = True
                active_distance = left_distance

            elif right_in_range:
                wrong_trigger = True

        elif expected_sensor == "right":

            if right_in_range:
                correct_trigger = True
                active_distance = right_distance

            elif left_in_range:
                wrong_trigger = True

        if (correct_trigger and current_time - last_note_time > NOTE_COOLDOWN):
            volume = distance_to_volume(active_distance)
            kick_type = get_kick_from_acceleration(accel_mag)

            print(f"PLAY: {kick_type} | Volume: {volume:.2f}")
            play(kick_type, volume)

            last_note_time = current_time
            wrong_pose_start = None

        elif wrong_trigger:

            if wrong_pose_start is None:
                wrong_pose_start = current_time
            held_time = current_time - wrong_pose_start

            if (held_time > FALSE_TRIGGER_TIME and
                current_time - last_note_time > NOTE_COOLDOWN):

                print("FALSE NOTE")
                play("false", 0.6)
                
                last_note_time = current_time
                wrong_pose_start = None

        else:
            wrong_pose_start = None

        time.sleep(0.02)

    except KeyboardInterrupt:
        print("Stopping.")
        break

    except Exception as e:
        print("Error:", e)
        time.sleep(0.1)