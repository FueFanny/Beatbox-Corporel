import time
import board
import busio
import adafruit_bno055

#python3 -m venv bnoenv
#source bnoenv/bin/activate
#python script2.py

def init_sensor():
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_bno055.BNO055_I2C(i2c)
    return sensor


def check_chip(sensor):
    print("=== CHIP ID CHECK ===")
    try:
        chip_id = sensor._read_register(0x00)
    except Exception as e:
        print(f"[ERROR] Could not read chip ID: {e}")
        return False

    print(f"Chip ID: 0x{chip_id:02X}")

    if chip_id == 0xA0:
        print("[PASS] Correct chip detected\n")
        return True
    else:
        print("[FAIL] Unexpected chip ID (expected 0xA0)\n")
        return False


def check_self_test(sensor):
    print("=== SELF TEST ===")
    try:
        result = sensor._read_register(0x36)
    except Exception as e:
        print(f"[ERROR] Could not read self-test: {e}")
        return False

    print(f"Raw result: 0b{result:08b}")

    tests = {
        "Accelerometer": (result >> 0) & 1,
        "Magnetometer":  (result >> 1) & 1,
        "Gyroscope":     (result >> 2) & 1,
        "MCU":           (result >> 3) & 1,
    }

    all_ok = True
    for name, status in tests.items():
        if status:
            print(f"[PASS] {name}")
        else:
            print(f"[FAIL] {name}")
            all_ok = False

    print()
    return all_ok

def read_motion(sensor, samples=10, delay=0.5):
    print("=== LIVE DATA TEST ===")
    print("Move the sensor to verify changing values...\n")

    valid = True

    for i in range(samples):
        accel = sensor.acceleration
        gyro = sensor.gyro
        euler = sensor.euler

        print(f"[Sample {i+1}]")
        print(f"  Accelerometer / acceleration : {accel}")
        print(f"  Gyroscope / rotation  : {gyro}")
        print(f"  Euler / angles : {euler}")

        # Basic sanity check
        if accel is None or gyro is None:
            valid = False

        print("-" * 40)
        time.sleep(delay)

    if valid:
        print("[PASS] Sensor is producing live data\n")
    else:
        print("[WARN] Some readings were None (possible issue)\n")

    return valid

def check_calibration(sensor, duration=15):
    print("=== CALIBRATION STATUS ===")
    print("(Values range from 0 = uncalibrated to 3 = fully calibrated)\n")

    for _ in range(duration):
        sys, gyro, accel, mag = sensor.calibration_status
        print(f"SYS={sys} GYRO={gyro} ACCEL={accel} MAG={mag}")
        time.sleep(1)

    print("\n[INFO] Move sensor in figure-8 motion to improve calibration\n")

def main():
    print("\n=== BNO055 DIAGNOSTIC TOOL ===\n")

    sensor = init_sensor()

    if not check_chip(sensor):
        print("[STOP] Cannot communicate with sensor")
        return

    self_test_ok = check_self_test(sensor)
    motion_ok = read_motion(sensor)
    check_calibration(sensor)

    print("=== SUMMARY ===")
    print(f"Self-test: {'PASS' if self_test_ok else 'FAIL'}")
    print(f"Live data: {'PASS' if motion_ok else 'CHECK'}")

    if self_test_ok and motion_ok:
        print("\n[SUCCESS] Sensor appears fully operational")
    else:
        print("\n[ATTENTION] Sensor may have issues")

    print("\n=== END ===\n")


if __name__ == "__main__":
    main()
