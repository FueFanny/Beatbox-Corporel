import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("motion_log.csv")
time = df["time_since_start"]

# ACCELERATION MAGNITUDE

ax = df["linear_x"].fillna(0)
ay = df["linear_y"].fillna(0)
az = df["linear_z"].fillna(0)

accel_mag = np.sqrt(
    ax**2 +
    ay**2 +
    az**2
)

# TILT STATE FROM ROLL

TILT_THRESHOLD = 15

roll = df["roll"]

tilt_numeric = []

for r in roll:
    if pd.isna(r):
        tilt_numeric.append(1)      # center
    elif r > TILT_THRESHOLD:
        tilt_numeric.append(2)      # right
    elif r < -TILT_THRESHOLD:
        tilt_numeric.append(0)      # left
    else:
        tilt_numeric.append(1)      # center

fig, (ax1, ax2) = plt.subplots(
    2,
    1,
    figsize=(12, 8),
    sharex=True
)

# GRAPH 1 : ACCELERATION

ax1.plot(
    time,
    accel_mag,
    linewidth=2
)

ax1.set_ylabel("Acceleration (m/s²)")
ax1.set_title("Acceleration Magnitude vs Time")
ax1.grid(True)

# GRAPH 2 : TILT

ax2.step(
    time,
    tilt_numeric,
    where="post",
    linewidth=2
)

ax2.set_yticks([0, 1, 2])
ax2.set_yticklabels([
    "Left",
    "Center",
    "Right"
])

ax2.set_ylabel("Tilt")
ax2.set_xlabel("Time (s)")
ax2.set_title("IMU Orientation vs Time")
ax2.grid(True)

plt.tight_layout()
plt.show()