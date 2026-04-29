import cv2

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
    display = frame.copy()

    for c in contours:
        area = cv2.contourArea(c)

        if area < 2000:
            continue

        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cx, cy = x + w // 2, y + h // 2
        cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow("Motion Tracker", display)
    cv2.imshow("Mask", fgmask)

    # Exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
