import pygame
import numpy
import math
import time

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
    # devant (50)
    {"name": "Do",  "center": (30,  0, 50), "radius": 20},
    {"name": "Re",  "center": (0,  30, 50), "radius": 20},
    {"name": "Mi",  "center": (-30, 0, 50), "radius": 20},

    # bas (30)
    {"name": "Fa",  "center": (20, -30, 30), "radius": 20},
    {"name": "Sol", "center": (-20, -30, 30), "radius": 20},

    # haut (70–80)
    {"name": "La",  "center": (20, 0, 75), "radius": 20},
    {"name": "Si",  "center": (-20, 0, 75), "radius": 20},
]


def in_playable_space(x, y, z):
    # derrière
    if z < 0:
        return False

    # trop bas
    if z < -20:
        return False

    # trop loin
    if math.sqrt(x*x + y*y + z*z) > 70:
        return False

    return True

def get_active_zone(x, y, z):
    if not in_playable_space(x, y, z):
        return None, 0

    for zone in zones:
        cx, cy, cz = zone["center"]

        d = math.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)

        if d <= zone["radius"]:
            volume = 1 - (d / zone["radius"])
            return zone["name"], volume

    return None, 0


x, y, z = 0.0, 0.0, 0.0

active_zone = None
last_play_time = 0
repeat_interval = 1.0

while True:

    note, volume = get_active_zone(x, y, z)
    now = time.time()

    print(f"Position: {x:.1f}, {y:.1f}, {z:.1f}")

    if note is not None:

        if note != active_zone:
            active_zone = note
            last_play_time = 0  # immediate trigger

        if now - last_play_time > repeat_interval:
            pitch = NOTES[note]
            sound = Piano(volume, pitch, 0.4)
            channel.play(sound)

            print(f"Playing: {note}")

            last_play_time = now

    else:
        active_zone = None


    # faux mouvements

    x += 3
    y = 30 * math.sin(pygame.time.get_ticks() * 0.002)
    z = 50 + 30 * math.sin(pygame.time.get_ticks() * 0.001)

    if x > 60:
        x = -60

    pygame.time.delay(10)

#---------------test

"""
melody = [
    ("Mi", 0.4), ("Mi", 0.4), ("Fa", 0.4), ("Sol", 0.4),
    ("Sol", 0.4), ("Fa", 0.4), ("Mi", 0.4), ("Re", 0.4),
    ("Do", 0.4), ("Do", 0.4), ("Re", 0.4), ("Mi", 0.4),
    ("Mi", 0.6), ("Re", 0.2), ("Re", 0.8)
]

zone_lookup = {zone["name"]: zone["center"] for zone in zones}

movement_sequence = []

for note, duration in melody:
    cx, cy, cz = zone_lookup[note]

    movement_sequence.append({
        "note": note,
        "target": (cx, cy, cz),
        "duration": duration
    })


x, y, z = 0.0, 0.0, 0.0

current_index = 0
note_start_time = time.time()
first_note, first_duration = melody[0]
channel.play(Piano(0.8, NOTES[first_note], first_duration))

while current_index < len(movement_sequence):

    current = movement_sequence[current_index]
    target = current["target"]
    duration = current["duration"]
    note_name = current["note"]

    cx, cy, cz = target

    speed = 0.15
    x += (cx - x) * speed
    y += (cy - y) * speed
    z += (cz - z) * speed

    print(f"Pos: {x:.1f}, {y:.1f}, {z:.1f} | Target: {note_name}")

    if time.time() - note_start_time > duration:
        current_index += 1
        if current_index >= len(movement_sequence):
            break

        next_note, next_duration = melody[current_index]
        channel.play(Piano(0.8, NOTES[next_note], next_duration))

        note_start_time = time.time()

    pygame.time.delay(20)

print("Melody finished")

"""