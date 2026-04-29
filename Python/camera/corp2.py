import cv2
import numpy as np
import tensorflow as tf
import time

interpreter = tf.lite.Interpreter(model_path="movenet_lightning.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 192)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)

prev_keypoints = None
movement_threshold = 0.03

# Body part names (MoveNet order)
parts = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

def detect_pose(frame):
    img = cv2.resize(frame, (192, 192))
    img = np.expand_dims(img, axis=0).astype(np.uint8)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return keypoints

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    keypoints = detect_pose(frame)

    h, w = frame.shape[:2]

    moving_parts = []

    for i, kp in enumerate(keypoints):

        y, x, conf = kp

        if conf < 0.3:
            continue

        px = int(x * w)
        py = int(y * h)

        cv2.circle(display, (px, py), 4, (0, 255, 0), -1)
        cv2.putText(display, parts[i], (px + 5, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if prev_keypoints is not None:

            py0, px0, c0 = prev_keypoints[i]

            if c0 > 0.3:

                dx = abs(x - px0)
                dy = abs(y - py0)

                if dx + dy > movement_threshold:
                    moving_parts.append(parts[i])

                    cv2.circle(display, (px, py), 8, (0, 0, 255), -1)

    if moving_parts:
        text = "Moving: " + ", ".join(set(moving_parts))
        cv2.putText(display, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

    prev_keypoints = keypoints

    cv2.imshow("MoveNet Body Tracker", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()