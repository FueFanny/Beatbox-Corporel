import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
import threading

interpreter = tflite.Interpreter(
    model_path="/home/alice/movenet_lightning.tflite",
    num_threads=4
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
cap.set(cv2.CAP_PROP_FPS, 30)

fgbg = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=25,
    detectShadows=False
)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

parts = [
    "head",
    "neck",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

# MoveNet indices we still use:
# 0 nose
# 5 left_shoulder, 6 right_shoulder
# 11 left_hip, 12 right_hip
# 7/8 elbows, 9/10 wrists, etc.

def simplify_keypoints(kp):
    """Convert MoveNet 17 points → 13 simplified points"""

    nose = kp[0]

    ls = kp[5]
    rs = kp[6]

    lh = kp[11]
    rh = kp[12]

    left_elbow = kp[7]
    right_elbow = kp[8]
    left_wrist = kp[9]
    right_wrist = kp[10]
    left_knee = kp[13]
    right_knee = kp[14]
    left_ankle = kp[15]
    right_ankle = kp[16]

    # head = nose
    head = nose

    # neck = midpoint of shoulders
    neck = [
        (ls[0] + rs[0]) / 2,
        (ls[1] + rs[1]) / 2,
        min(ls[2], rs[2])
    ]

    return np.array([
        head,
        neck,
        ls, rs,
        left_elbow, right_elbow,
        left_wrist, right_wrist,
        lh, rh,
        left_knee, right_knee,
        left_ankle, right_ankle
    ])


latest_frame = None
latest_keypoints = None
lock = threading.Lock()


def pose_worker():
    global latest_frame, latest_keypoints

    while True:
        if latest_frame is None:
            time.sleep(0.005)
            continue

        with lock:
            frame = latest_frame.copy()

        img = cv2.resize(frame, (192, 192))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        kp = interpreter.get_tensor(output_details[0]['index'])[0][0]
        kp = simplify_keypoints(kp)

        with lock:
            latest_keypoints = kp


threading.Thread(target=pose_worker, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    with lock:
        latest_frame = frame
        keypoints = latest_keypoints

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=2)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    moving_parts = []

    if keypoints is not None:
        h, w = frame.shape[:2]

        for i, kp in enumerate(keypoints):
            y, x, conf = kp
            if conf < 0.3:
                continue

            px, py = int(x * w), int(y * h)

            cv2.circle(display, (px, py), 4, (255, 255, 255), -1)

            if 0 <= px < w and 0 <= py < h and fgmask[py, px] > 0:
                moving_parts.append(parts[i])
                cv2.circle(display, (px, py), 7, (0, 0, 255), -1)
                cv2.putText(display, parts[i],
                            (px + 5, py),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255), 1)

    if moving_parts:
        cv2.putText(display,
                    "Moving: " + ", ".join(set(moving_parts)),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255), 2)

    cv2.imshow("Simplified Real-Time Body Motion", display)
    cv2.imshow("Mask", fgmask)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
