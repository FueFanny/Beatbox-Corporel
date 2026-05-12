import time
import math
import board
import busio
import pygame

from adafruit_bno055 import BNO055_I2C
from gpiozero import DistanceSensor

i2c = busio.I2C(board.SCL, board.SDA)
imu = BNO055_I2C(i2c)

ultrasonic = DistanceSensor(
    echo=24,
    trigger=23,
    max_distance=2.0
)

pygame.mixer.init()

sounds = {
    "kick_soft": pygame.mixer.Sound("sounds/soft/kick.wav"),
    "kick_medium": pygame.mixer.Sound("sounds/medium/kick.wav"),
    "kick_hard": pygame.mixer.Sound("sounds/hard/kick.wav"),
    "false": pygame.mixer.Sound("sounds/false/false.wav")
}

# tilt detection
TILT_THRESHOLD = 15
GROUND_THRESHOLD = 0.55
MIN_DISTANCE = 0.15

NOTE_COOLDOWN = 0.25
FALSE_TRIGGER_TIME = 0.45

MEDIUM_ACCEL = 2.0
HARD_ACCEL = 4.0

last_note_time = 0
right_tilt_start = None

previous_accel_mag = 0

def clamp(val, low, high):
    return max(low, min(high, val))

def distance_to_volume(distance):
    normalized = 1.0 - (
        (distance - MIN_DISTANCE) /
        (GROUND_THRESHOLD - MIN_DISTANCE)
    )
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

print("Interactive movement music system started.")

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

        distance = ultrasonic.distance
        distance_cm = distance * 100

        hand_in_range = (MIN_DISTANCE <= distance <= GROUND_THRESHOLD)

        ax, ay, az = acceleration
        accel_mag = math.sqrt(ax * ax + ay * ay + az * az)

        print(
            f"Tilt: {tilt} | "
            f"Roll: {roll:.1f} | "
            f"Distance: {distance_cm:.1f} cm | "
            f"Accel: {accel_mag:.2f}"
        )

        current_time = time.time()

        valid_pose = (tilt == "left" and hand_in_range)
        cheating_pose = (tilt == "right" and hand_in_range)


        if (valid_pose and
            current_time - last_note_time > NOTE_COOLDOWN):
            volume = distance_to_volume(distance)
            kick_type = get_kick_from_acceleration(accel_mag)

            print(f"PLAY: {kick_type} | Volume: {volume:.2f}")

            play(kick_type, volume)
            last_note_time = current_time

        elif cheating_pose:
            if right_tilt_start is None:
                right_tilt_start = current_time
            held_time = current_time - right_tilt_start
            
            if (held_time > FALSE_TRIGGER_TIME and
                current_time - last_note_time > NOTE_COOLDOWN):
                print("FALSE NOTE")
                play("false", 0.6)
                last_note_time = current_time
                right_tilt_start = None

        else:
            right_tilt_start = None
        time.sleep(0.02)

    except KeyboardInterrupt:
        print("Stopping.")
        break

    except Exception as e:
        print("Error:", e)
        time.sleep(0.1)
