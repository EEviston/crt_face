import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

ghost = None

# --- scanline settings ---
line_spacing = 4        # every 4 pixels
line_intensity = 0.9    # 0 = black line, 1 = unchanged

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # ghosting
    if ghost is None:
        ghost = frame.astype(float)

    alpha = 0.5
    ghost = cv2.addWeighted(frame.astype(float), alpha, ghost, 1 - alpha, 0)
    display_frame = ghost.astype('uint8')

    # face rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (10, 255, 30), 2)

    # scanlines
    for y in range(0, display_frame.shape[0], line_spacing):
        display_frame[y:y+1, :, :] = (display_frame[y:y+1, :, :] * line_intensity).astype('uint8')

    cv2.imshow("", display_frame)

    if cv2.waitKey(1) & 0xFF == 27: # esc key
        break

cap.release()
cv2.destroyAllWindows()
