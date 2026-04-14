import pygame
import numpy
import math

#init
pygame.mixer.init(frequency=44100, size=-16, channels=2)
sample_rate = 44100

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


zones = [
    {"center": (50, 0, 50), "radius": 40, "note": 261.63},  # C4
    {"center": (-50, 0, 50), "radius": 40, "note": 293.66}, # D4
    {"center": (0, 50, 100), "radius": 40, "note": 329.63}, # E4
    {"center": (0, -50, 100), "radius": 40, "note": 349.23} # F4
]

dead_radius = 30  # silence near feet

def get_active_zone(x, y, z):
    # distance from origin
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

while True:

    zone, volume = get_active_zone(x, y, z)

    if zone is not None:
        note = zone["note"]

        if note != current_note:
            duration = 0.3

            sound = Piano(volume, note, duration)
            channel.play(sound)

            current_note = note
    else:
        current_note = None


    x += 5
    y += 2
    z = abs(100 * math.sin(pygame.time.get_ticks() * 0.001))

    if x > 100:
        x = -100
    if y > 100:
        y = -100

    pygame.time.delay(100)