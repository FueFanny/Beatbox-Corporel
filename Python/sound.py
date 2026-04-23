import pygame
import numpy
import math

#init
pygame.mixer.init(frequency=44100, size=-16, channels=2)
sample_rate = 44100
channel = pygame.mixer.Channel(0)

def Piano(volume, pitch, duration):
    t = numpy.linspace(0, duration, int(sample_rate * duration), False)

    wave = (
        numpy.sin(2 * numpy.pi * pitch * t) +
        0.5 * numpy.sin(2 * numpy.pi * 2 * pitch * t) +
        0.25 * numpy.sin(2 * numpy.pi * 3 * pitch * t)
    )

    attack = int(0.02 * sample_rate)
    envelope = numpy.ones_like(t)

    envelope[:attack] = numpy.linspace(0, 1, attack)
    envelope[attack:] = numpy.exp(-3 * (t[attack:]))

    wave = wave / numpy.max(numpy.abs(wave))
    wave = wave * envelope * volume * 0.5

    audio = (wave * 32767).astype(numpy.int16)
    stereo = numpy.column_stack((audio, audio))

    return pygame.sndarray.make_sound(stereo)

NOTES = {
    "Do": 261.63,
    "Re": 293.66,
    "Mi": 329.63,
    "Fa": 349.23,
    "Sol": 392.00,
    "La": 440.00,
    "Si": 493.88
}

zones = [
    {"center": (50, 0, 50), "radius": 40, "note": 261.63},  # C4
    {"center": (-50, 0, 50), "radius": 40, "note": 293.66}, # D4
    {"center": (0, 50, 100), "radius": 40, "note": 329.63}, # E4
    {"center": (0, -50, 100), "radius": 40, "note": 349.23} # F4
]


melody = [
    ("Mi", 0.4), 
    ("Mi", 0.4), 
    ("Fa", 0.4), 
    ("Sol", 0.4),
    ("Sol", 0.4), 
    ("Fa", 0.4), 
    ("Mi", 0.4), 
    ("Re", 0.4),
    ("Do", 0.4), 
    ("Do", 0.4), 
    ("Re", 0.4), 
    ("Mi", 0.4),
    ("Mi", 0.6), 
    ("Re", 0.2), 
    ("Re", 0.8)
]

dead_radius = 30  # silence near feet

def get_active_zone(x, y, z):
    d0 = math.sqrt(x*x + y*y + z*z)

    if d0 < dead_radius:
        return None, 0

    closest_zone = None
    min_dist = float("inf")

    for zone in zones:
        cx, cy, cz = zone["center"]

        d = math.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)

        if d <= zone["radius"] and d < min_dist:
            min_dist = d
            closest_zone = zone

    if closest_zone:
        volume = 1 - (min_dist / closest_zone["radius"])
        return closest_zone, volume

    return None, 0


x, y, z = 0.0, 0.0, 0.0

current_note = None
channel = pygame.mixer.Channel(0)
zone_sequence = [zone["name"] for zone in zones] * 3
zone_index = 0

import time

current_note = None
current_index = 0

last_trigger_time = 0
cooldown = 0.4

while current_index < len(melody):

    note, volume, distance = get_active_zone(x, y, z)

    current_time = time.time()

    print(f"Position: x={x:.1f}, y={y:.1f}, z={z:.1f}")

    if (note is not None and note != current_note and current_time - last_trigger_time > cooldown):

        melody_note, duration = melody[current_index]

        pitch = NOTES[melody_note]
        sound = Piano(0.5, pitch, duration)
        channel.play(sound)

        print(f"Triggered: {melody_note}")

        current_index += 1
        current_note = note
        last_trigger_time = current_time

    elif note is None:
        current_note = None

    x += 5
    y += 2
    z = abs(100 * math.sin(pygame.time.get_ticks() * 0.001))

    if x > 100:
        x = -100
    if y > 100:
        y = -100

    pygame.time.delay(10)

"""
for note_name, duration in melody:

    if note_name is not None:
        pitch = NOTES[note_name]
        sound = Piano(0.5, pitch, duration)
        channel.play(sound)

        print(f"Playing: {note_name}")

    else:
        print("Rest")

    pygame.time.delay(int(duration * 1000))

"""
