1. Replace import
# ❌ REMOVE
from gpiozero import DistanceSensor

# ✅ ADD
import pigpio
2. Replace ultrasonic initialization block
# =========================================================
# ULTRASONIC SENSORS (pigpio)
# =========================================================

TRIG_LEFT = 23
ECHO_LEFT = 24

TRIG_RIGHT = 27
ECHO_RIGHT = 17

pi = pigpio.pi()

if not pi.connected:
    print("ERROR: pigpio daemon not running")
    sys.exit(1)

pi.set_mode(TRIG_LEFT, pigpio.OUTPUT)
pi.set_mode(ECHO_LEFT, pigpio.INPUT)

pi.set_mode(TRIG_RIGHT, pigpio.OUTPUT)
pi.set_mode(ECHO_RIGHT, pigpio.INPUT)

pi.write(TRIG_LEFT, 0)
pi.write(TRIG_RIGHT, 0)

time.sleep(0.1)
3. Add this function (near your helpers)
def read_distance(trig, echo):
    pi.gpio_trigger(trig, 10)

    start = time.time()
    timeout = start + 0.02

    # wait HIGH
    while pi.read(echo) == 0:
        if time.time() > timeout:
            return None
        start = time.time()

    # wait LOW
    while pi.read(echo) == 1:
        if time.time() > timeout:
            return None
        end = time.time()

    return (end - start) * 343 / 2
4. Replace ONLY this part in your main loop
❌ REMOVE this block:
if loop_counter % 2 == 0:
    raw_left_distance = left_sensor.distance
else:
    raw_right_distance = right_sensor.distance
time.sleep(0.067)
✅ REPLACE with:
# pigpio sequential read (stable)

raw_left_distance = read_distance(TRIG_LEFT, ECHO_LEFT)
time.sleep(0.03)

raw_right_distance = read_distance(TRIG_RIGHT, ECHO_RIGHT)
time.sleep(0.03)

print("LEFT:", raw_left_distance, "RIGHT:", raw_right_distance)
5. Add cleanup (IMPORTANT)

Inside clean_exit():

pi.stop()
✔ What stays EXACTLY the same
