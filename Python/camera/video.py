import os
import cv2
import pandas as pd
import numpy as np
from math import sqrt

CSV_FILE = "motion_log.csv"
IMAGE_DIR = "captures"
OUTPUT_VIDEO = "motion_reconstruction.mp4"

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30

#taille image
SMALL_SCALE = 0.45
MEDIUM_SCALE = 0.70
LARGE_SCALE = 1.00

#acceleration
LOW_THRESHOLD = 2.0
HIGH_THRESHOLD = 6.0

BACKGROUND_COLOR = (0, 0, 0)


df = pd.read_csv(CSV_FILE)

if len(df) == 0:
    raise Exception("CSV is empty.")
df = df.sort_values("time_since_start").reset_index(drop=True)
durations = []

for i in range(len(df)):
    current_time = df.loc[i, "time_since_start"]

    if i < len(df) - 1:
        next_time = df.loc[i + 1, "time_since_start"]
        duration = max(0.1, next_time - current_time)
    else:
        duration = 1.5

    durations.append(duration)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO,fourcc,FPS,(VIDEO_WIDTH, VIDEO_HEIGHT))

def get_acceleration_magnitude(row):
    ax = row.get("linear_x", 0)
    ay = row.get("linear_y", 0)
    az = row.get("linear_z", 0)
    ax = 0 if pd.isna(ax) else ax
    ay = 0 if pd.isna(ay) else ay
    az = 0 if pd.isna(az) else az

    magnitude = sqrt(ax**2 + ay**2 + az**2)

    return magnitude


def get_scale(accel_mag):
    if accel_mag < LOW_THRESHOLD:
        return SMALL_SCALE

    elif accel_mag < HIGH_THRESHOLD:
        return MEDIUM_SCALE
    else:
        return LARGE_SCALE


def place_image(canvas, image, scale):
    h, w = image.shape[:2]

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h))

    x = (VIDEO_WIDTH - new_w) // 2
    y = (VIDEO_HEIGHT - new_h) // 2

    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas

for i, row in df.iterrows():
    filename = row["image_file"]
    filepath = os.path.join(IMAGE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"manque image")
        continue
    image = cv2.imread(filepath)
    if image is None:
        print(f"pas pu trouver image")
        continue
    accel_mag = get_acceleration_magnitude(row)
    scale = get_scale(accel_mag)
    duration = durations[i]
    frame_count = int(duration * FPS)

    print(
        f"{filename} | "
        f"accel={accel_mag:.2f} | "
        f"scale={scale} | "
        f"duration={duration:.2f}s"
    )
    for _ in range(frame_count):
        canvas = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3),dtype=np.uint8)
        canvas[:] = BACKGROUND_COLOR
        final_frame = place_image(canvas, image, scale)
        text = f"Accel: {accel_mag:.2f}"

        cv2.putText(final_frame,text,(30, VIDEO_HEIGHT - 30),cv2.FONT_HERSHEY_SIMPLEX,1,(255, 255, 255),2)
        writer.write(final_frame)

writer.release()
print(f"\nSuvegardé: {OUTPUT_VIDEO}")