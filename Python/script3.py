import time
import board
import busio
import adafruit_bno055
import os

def init_sensor():
    i2c = busio.I2C(board.SCL, board.SDA)
    return adafruit_bno055.BNO055_I2C(i2c)

def clear():
    os.system("clear")

def main():
    sensor = init_sensor()

    print("Starting live BNO055 stream (CTRL+C to stop)...")
    time.sleep(2)

    while True:
        accel = sensor.acceleration
        gyro = sensor.gyro
        euler = sensor.euler
        calib = sensor.calibration_status

        clear()

        print("=== LIVE BNO055 DATA ===\n")

        print("Acceleration (m/s²):")
        print(f"  X: {accel[0]:6.2f}  Y: {accel[1]:6.2f}  Z: {accel[2]:6.2f}\n")

        print("Gyroscope (rad/s):")
        print(f"  X: {gyro[0]:6.4f}  Y: {gyro[1]:6.4f}  Z: {gyro[2]:6.4f}\n")

        print("Euler Angles (degrees):")
        print(f"  Heading: {euler[0]:6.2f}")
        print(f"  Roll   : {euler[1]:6.2f}")
        print(f"  Pitch  : {euler[2]:6.2f}\n")

        sys, g, a, m = calib
        print("Calibration Status (0–3):")
        print(f"  SYS={sys}  GYRO={g}  ACCEL={a}  MAG={m}\n")

        print("Move sensor to see changes...")
        print("Keep it still to calibrate gyro")

        time.sleep(0.2)  # adjust refresh rate here

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped by user")
