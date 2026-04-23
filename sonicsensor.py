import RPi.GPIO as GPIO
import time
import pygame
import numpy as np

TRIG = 23
ECHO = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)
time.sleep(2)

pygame.mixer.init(frequency=44100, size=-16, channels=2)
sample_rate = 44100
channel = pygame.mixer.Channel(0)

def make_sound(freq, duration=0.3, volume=0.8):
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    wave = np.sin(2 * np.pi * freq * t)

    envelope = np.exp(-4 * t)
    wave *= envelope

    audio = (wave * volume * 32767).astype(np.int16)
    stereo = np.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo)

sounds = [
    make_sound(220),  # 0-20 cm
    make_sound(330),  # 20-40
    make_sound(440),  # 40-60
    make_sound(550),  # 60-80
    make_sound(660)   # 80-100
]

def measure_distance():
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.time() + 0.02
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return None

    timeout = time.time() + 0.02
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return None

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return round(distance, 2)

last_zone = None
last_time = 0
cooldown = 0.5  # seconds

def get_zone(distance):
    if distance is None or distance > 100 or distance < 0:
        return None

    return int(distance // 20)  # 0 to 4

try:
    print("Ultrasonic sound system started")

    while True:
        dist = measure_distance()
        zone = get_zone(dist)
        now = time.time()

        print(f"Distance: {dist} cm | Zone: {zone}")

        if zone is not None:

            if zone != last_zone:
                channel.play(sounds[zone])
                print(f"New zone sound: {zone}")
                last_zone = zone
                last_time = now

            elif now - last_time > cooldown:
                channel.play(sounds[zone])
                print(f"Repeat zone sound: {zone}")
                last_time = now

        else:
            last_zone = None

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopped")
    GPIO.cleanup()
    pygame.quit()
