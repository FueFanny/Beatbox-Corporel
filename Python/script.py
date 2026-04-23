import RPi.GPIO as GPIO
import time

TRIG = 23
ECHO = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

GPIO.output(TRIG, False)
time.sleep(2)

print("Ultrasonic sensor test started")

def measure_distance():
    # Send trigger pulse
    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    # Wait for echo to go HIGH
    timeout = time.time() + 0.02  # 20 ms timeout
    while GPIO.input(ECHO) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return "ERROR: No echo received (ECHO stayed LOW)"

    # Wait for echo to go LOW
    timeout = time.time() + 0.02
    while GPIO.input(ECHO) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return "ERROR: Echo never ended (ECHO stuck HIGH)"

    # Calculate distance
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return round(distance, 2)


try:
    while True:
        result = measure_distance()

        if isinstance(result, str):
            print(result)
        else:
            print(f"Distance: {result} cm")

        time.sleep(1)

except KeyboardInterrupt:
    print("\nTest stopped")
    GPIO.cleanup()
