import os
import cv2
import pandas as pd
import numpy as np
from math import sqrt
from pydub import AudioSegment
import subprocess

CSV_FILE = "motion_log.csv"
IMAGE_DIR = "captures"

TEMP_VIDEO = "temp_video.mp4"
AUDIO_TRACK = "audio_track.wav"
OUTPUT_VIDEO = "motion_layers.mp4"

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30

BACKGROUND_COLOR = (0, 0, 0)

FADE_MULTIPLIER = 0.94

SMALL_SCALE = 0.45
MEDIUM_SCALE = 0.70
LARGE_SCALE = 1.00

LOW_ACCEL = 1.3
HIGH_ACCEL = 2.8

SOUND_FILES = {
    "kick_soft": "sounds/soft/kick.wav",
    "kick_medium": "sounds/medium/kick.wav",
    "kick_hard": "sounds/hard/kick.wav",
    "false": "sounds/false/false.wav"
}

# =========================================================
# LOAD CSV
# =========================================================

df = pd.read_csv(CSV_FILE)

if len(df) == 0:
    raise Exception("CSV is empty.")

df = df.sort_values(
    "time_since_start"
).reset_index(drop=True)

# =========================================================
# VIDEO WRITER
# =========================================================

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

writer = cv2.VideoWriter(
    TEMP_VIDEO,
    fourcc,
    FPS,
    (VIDEO_WIDTH, VIDEO_HEIGHT)
)

# =========================================================
# HELPERS
# =========================================================

def safe(v, default=0):

    if pd.isna(v):
        return default

    return v


def get_acceleration_magnitude(row):

    ax = safe(row.get("linear_x"))
    ay = safe(row.get("linear_y"))
    az = safe(row.get("linear_z"))

    return sqrt(ax**2 + ay**2 + az**2)


def get_scale(accel_mag):

    if accel_mag < LOW_ACCEL:
        return SMALL_SCALE

    elif accel_mag < HIGH_ACCEL:
        return MEDIUM_SCALE

    else:
        return LARGE_SCALE


def resize_image(image, scale):

    h, w = image.shape[:2]

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(
        image,
        (new_w, new_h)
    )


def rotate_image(image, angle):

    h, w = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(
        image,
        matrix,
        (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    return rotated


def overlay_image(background, image, x, y):

    h, w = image.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if x >= bg_w or y >= bg_h:
        return

    if x + w <= 0 or y + h <= 0:
        return

    x1 = max(x, 0)
    y1 = max(y, 0)

    x2 = min(x + w, bg_w)
    y2 = min(y + h, bg_h)

    img_x1 = x1 - x
    img_y1 = y1 - y

    img_x2 = img_x1 + (x2 - x1)
    img_y2 = img_y1 + (y2 - y1)

    roi = background[y1:y2, x1:x2]

    img_crop = image[
        img_y1:img_y2,
        img_x1:img_x2
    ]

    mask = np.any(img_crop > 0, axis=2)

    roi[mask] = img_crop[mask]

    background[y1:y2, x1:x2] = roi

# =========================================================
# BUILD EVENTS
# =========================================================

events = []

for _, row in df.iterrows():

    filename = row.get("image_file")

    if pd.isna(filename):
        continue

    filepath = os.path.join(
        IMAGE_DIR,
        filename
    )

    if not os.path.exists(filepath):
        continue

    image = cv2.imread(filepath)

    if image is None:
        continue

    accel_mag = get_acceleration_magnitude(row)

    scale = get_scale(accel_mag)

    image = resize_image(
        image,
        scale
    )

    roll = safe(row.get("roll"))

    rotation_angle = roll * 1.2

    image = rotate_image(
        image,
        rotation_angle
    )

    timestamp = safe(
        row.get("time_since_start")
    )

    sound_name = row.get("sound_played")

    left_distance = safe(
        row.get("left_distance_cm")
    )

    right_distance = safe(
        row.get("right_distance_cm")
    )

    tilt = row.get("tilt")

    events.append({
        "time": timestamp,
        "image": image,
        "accel": accel_mag,
        "left_distance": left_distance,
        "right_distance": right_distance,
        "sound": sound_name,
        "tilt": tilt
    })

if len(events) == 0:
    raise Exception("No valid image events.")

# =========================================================
# TIMELINE
# =========================================================

total_duration = events[-1]["time"] + 2.0

total_frames = int(
    total_duration * FPS
)

active_layers = []
event_index = 0

canvas = np.zeros(
    (VIDEO_HEIGHT, VIDEO_WIDTH, 3),
    dtype=np.float32
)

# =========================================================
# RENDER VIDEO
# =========================================================

for frame_idx in range(total_frames):

    current_time = frame_idx / FPS

    canvas *= FADE_MULTIPLIER

    while (
        event_index < len(events)
        and events[event_index]["time"] <= current_time
    ):

        event = events[event_index]

        img = event["image"]

        h, w = img.shape[:2]

        x = (VIDEO_WIDTH - w) // 2
        y = (VIDEO_HEIGHT - h) // 2

        active_layers.append({
            "image": img.astype(np.float32),
            "x": x,
            "y": y,
            "alpha": 1.0,
            "sound": event["sound"],
            "left_distance": event["left_distance"],
            "right_distance": event["right_distance"],
            "accel": event["accel"],
            "tilt": event["tilt"]
        })

        event_index += 1

    new_layers = []

    frame = canvas.copy()

    for layer in active_layers:

        img = layer["image"]

        alpha = layer["alpha"]

        blended = img * alpha

        overlay_image(
            frame,
            blended.astype(np.uint8),
            layer["x"],
            layer["y"]
        )

        layer["alpha"] *= 0.965

        if layer["alpha"] > 0.03:
            new_layers.append(layer)

    active_layers = new_layers

    canvas = frame

    output = np.clip(
        canvas,
        0,
        255
    ).astype(np.uint8)

    # =====================================================
    # UI TEXT
    # =====================================================

    cv2.putText(
        output,
        f"Time: {current_time:.2f}s",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    if active_layers:

        latest = active_layers[-1]

        cv2.putText(
            output,
            f"Sound: {latest['sound']}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            f"Left Distance: {latest['left_distance']:.1f} cm",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 100, 255),
            2
        )

        cv2.putText(
            output,
            f"Right Distance: {latest['right_distance']:.1f} cm",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 200, 255),
            2
        )

        cv2.putText(
            output,
            f"Accel: {latest['accel']:.2f}",
            (30, 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"Tilt: {latest['tilt']}",
            (30, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (150, 255, 150),
            2
        )

    writer.write(output)

    print(
        f"Rendering frame {frame_idx+1}/{total_frames}",
        end="\r"
    )

# =========================================================
# SAVE VIDEO
# =========================================================

writer.release()

# =========================================================
# AUDIO TRACK
# =========================================================

print("\nGenerating audio track...")

total_duration_ms = int(
    total_duration * 1000
) + 2000

final_audio = AudioSegment.silent(
    duration=total_duration_ms
)

last_sound = None

for _, row in df.iterrows():

    sound_name = row.get("sound_played")

    if pd.isna(sound_name):
        last_sound = None
        continue

    # only trigger when sound changes
    if sound_name == last_sound:
        continue

    last_sound = sound_name

    if sound_name not in SOUND_FILES:
        continue

    sound_path = SOUND_FILES[sound_name]

    if not os.path.exists(sound_path):
        continue

    sound = AudioSegment.from_wav(
        sound_path
    )

    # optional shortening
    sound = sound[:350]

    sound_time_ms = int(
        safe(row.get("time_since_start")) * 1000
    )

    final_audio = final_audio.overlay(
        sound,
        position=sound_time_ms
    )

    print(
        f"Added sound: {sound_name} "
        f"at {sound_time_ms}ms"
    )

# =========================================================
# EXPORT AUDIO
# =========================================================

final_audio.export(
    AUDIO_TRACK,
    format="wav"
)

# =========================================================
# FINAL VIDEO
# =========================================================

print("Merging video + audio...")

subprocess.run([
    "ffmpeg",
    "-y",
    "-i", TEMP_VIDEO,
    "-i", AUDIO_TRACK,
    "-c:v", "copy",
    "-c:a", "aac",
    OUTPUT_VIDEO
])

print(f"\nSaved: {OUTPUT_VIDEO}")