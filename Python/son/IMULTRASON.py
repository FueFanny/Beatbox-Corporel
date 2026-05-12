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

# Make kick sounds strong
sounds["kick_soft"].set_volume(1.0)
sounds["kick_medium"].set_volume(1.0)
sounds["kick_hard"].set_volume(1.0)

# Make false sound quieter
sounds["false"].set_volume(0.18)

TILT_THRESHOLD = 15

# Start interaction at 40 cm
START_DISTANCE = 0.40

# Closest useful distance
MIN_DISTANCE = 0.03

# Smooth ultrasonic jitter
SMOOTHING = 0.75

# Less spammy sound switching
KICK_CHANGE_COOLDOWN = 0.65

# False trigger timing
FALSE_TRIGGER_TIME = 0.55

# MUCH more sensitive thresholds
MEDIUM_ACCEL = 1.3
HARD_ACCEL = 2.8

# Keep strongest motion entering zone
MOTION_MEMORY_TIME = 0.45

MIN_PLAY_TIME = 0.4


filtered_distance = START_DISTANCE
last_play_time = 0
current_channel = None
current_sound_name = None

right_tilt_start = None

last_kick_change = 0

# stores strongest recent acceleration
peak_accel = 0
peak_accel_time = 0

def clamp(val, low, high):
    return max(low, min(high, val))

def distance_to_volume(distance):
    """
    Closer to ground = MUCH louder
    """

    normalized = 1.0 - (
        (distance - MIN_DISTANCE) /
        (START_DISTANCE - MIN_DISTANCE)
    )

    normalized = clamp(normalized, 0.0, 1.0)

    # aggressive curve
    boosted = normalized ** 0.35

    # never too quiet
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

    if current_channel:
        current_channel.stop()

    current_channel = None
    current_sound_name = None

def play_loop(sound_name, volume):

    global current_channel
    global current_sound_name
    global last_kick_change

    current_time = time.time()

    if sound_name not in sounds:
        return

    should_change = (
        current_sound_name != sound_name and
        current_time - last_kick_change > KICK_CHANGE_COOLDOWN
    )

    # start first sound
    if current_channel is None:

        current_channel = sounds[sound_name].play(loops=-1)

        current_sound_name = sound_name
        last_kick_change = current_time

    # only switch occasionally
    elif should_change:

        stop_sound()

        current_channel = sounds[sound_name].play(loops=-1)

        current_sound_name = sound_name
        last_kick_change = current_time

    # stronger volume response
    boosted_volume = min(volume * 2.2, 1.0)

    if current_channel:
        current_channel.set_volume(boosted_volume)


print("Interactive movement music system started.")

while True:

    try:

        euler = imu.euler
        acceleration = imu.linear_acceleration

        if euler is None or acceleration is None:
            continue

        yaw, roll, pitch = euler

        if roll is None:
            continue

        ax, ay, az = acceleration

        if ax is None or ay is None or az is None:
            continue


        tilt = "center"

        if roll > TILT_THRESHOLD:
            tilt = "right"

        elif roll < -TILT_THRESHOLD:
            tilt = "left"


        raw_distance = ultrasonic.distance

        if raw_distance is None:
            continue

        # smoothing
        filtered_distance = (
            SMOOTHING * filtered_distance +
            (1 - SMOOTHING) * raw_distance
        )

        distance_cm = filtered_distance * 100

        hand_in_range = (
            MIN_DISTANCE <= filtered_distance <= START_DISTANCE
        )

        accel_mag = math.sqrt(
            ax * ax +
            ay * ay +
            az * az
        )

        current_time = time.time()

        if accel_mag > peak_accel:
            peak_accel = accel_mag
            peak_accel_time = current_time

        # decay memory over time
        if current_time - peak_accel_time > MOTION_MEMORY_TIME:
            peak_accel *= 0.92

        effective_accel = max(accel_mag, peak_accel)

        print(
            f"Tilt: {tilt} | "
            f"Roll: {roll:.1f} | "
            f"Distance: {distance_cm:.1f} cm | "
            f"Accel: {accel_mag:.2f} | "
            f"Effective: {effective_accel:.2f}"
        )

        valid_pose = (
            tilt == "left" and
            hand_in_range
        )

        cheating_pose = (
            tilt == "right" and
            hand_in_range
        )

        if (valid_pose and current_time - last_play_time > MIN_PLAY_TIME):

            volume = distance_to_volume(filtered_distance)

            kick_type = get_kick_from_acceleration(
                effective_accel
            )

            print(
                f"PLAYING: {kick_type} | "
                f"Volume: {volume:.2f}"
            )

            play_loop(
                kick_type,
                volume
            )
            last_play_time = current_time

        else:
            stop_sound()

        if cheating_pose:

            if right_tilt_start is None:
                right_tilt_start = current_time

            held_time = (
                current_time - right_tilt_start
            )

            if held_time > FALSE_TRIGGER_TIME:

                print("FALSE NOTE")

                sounds["false"].play()

                right_tilt_start = None

        else:
            right_tilt_start = None

        time.sleep(0.025)

    except KeyboardInterrupt:

        print("Stopping.")

        stop_sound()

        break

    except Exception as e:

        print("Error:", e)

        time.sleep(0.1)
