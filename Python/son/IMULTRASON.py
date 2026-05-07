import time
import board
import busio
import pygame
from adafruit_bno055 import BNO055_I2C
from gpiozero import DistanceSensor

i2c = busio.I2C(board.SCL, board.SDA)
imu = BNO055_I2C(i2c)
ultrasonic = DistanceSensor(echo=24,trigger=23,max_distance=2.0)

pygame.mixer.init()

sounds = {
    "soft": pygame.mixer.Sound("sounds/soft/kick.wav"),
    "medium": pygame.mixer.Sound("sounds/medium/kick.wav"),
    "hard": pygame.mixer.Sound("sounds/hard/kick.wav"),
    "false": pygame.mixer.Sound("sounds/false/false.wav")
}

TILT_THRESHOLD = 15
GROUND_THRESHOLD = 0.35
NOTE_COOLDOWN = 0.25

last_note_time = 0

def get_sound(distance):

    # close to ground
    if distance < 0.10:
        return "hard"
    # medium reach
    elif distance < 0.20:
        return "medium"
    # slight reach
    else:
        return "soft"

def play(name):
    if name in sounds:
        sounds[name].play()

print("Interactive movement music system started.")

while True:
    try:
        euler = imu.euler
        if euler is None:
            continue
        yaw, roll, pitch = euler
        tilt = "center"

        if roll > TILT_THRESHOLD:
            tilt = "right"
        elif roll < -TILT_THRESHOLD:
            tilt = "left"

        distance = ultrasonic.distance
        distance_cm = distance * 100

        print(
            f"Tilt: {tilt} | "
            f"Roll: {roll:.1f} | "
            f"Distance: {distance_cm:.1f} cm"
        )
        
        current_time = time.time()
        hand_near_ground = distance < GROUND_THRESHOLD

        valid_pose = (tilt == "left" andhand_near_ground)
        cheating_pose = (tilt == "right" andhand_near_ground)

        if current_time - last_note_time > NOTE_COOLDOWN:
            if valid_pose:
                sound_type = get_sound(distance)
                print(f"PLAY: {sound_type}")
                play(sound_type)
                last_note_time = current_time

            elif cheating_pose:
                print("FALSE NOTE")
                play("false")
                last_note_time = current_time

        time.sleep(0.02)

    except KeyboardInterrupt:
        print("Stopping.")
        break

    except Exception as e:
        print("Error:", e)
        time.sleep(0.1)