import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

ghost = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ghosting
    if ghost is None:
        ghost = frame.astype(float)

    alpha = 0.2
    ghost = cv2.addWeighted(frame.astype(float), alpha, ghost, 1 - alpha, 0)
    display_frame = ghost.astype('uint8')

    # face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (10, 255, 30), 2)

    cv2.imshow("", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # esc key
        break

cap.release()
cv2.destroyAllWindows()
