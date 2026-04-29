import cv2
import time
import os

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

save_path = "captures"
os.makedirs(save_path, exist_ok=True)

cooldown = 1.0
last_capture_time = 0

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (480, 360))

    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=2)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    movement_detected = False

    for c in contours:
        if cv2.contourArea(c) > 2000:
            movement_detected = True
            break

    #SCREENSHOT
    current_time = time.time()

    if movement_detected and (current_time - last_capture_time > cooldown):
        filename = os.path.join(save_path, f"capture_{frame_id}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")

        last_capture_time = current_time
        frame_id += 1

    #DISPLAY
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", fgmask)

    # Exit with ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()