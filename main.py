import cv2
import numpy as np

from profiles import ProfileManager

CRT_WIDTH, CRT_HEIGHT = 720, 480
LINE_SPACING = 4
LINE_INTENSITY = 1.0
GHOST_ALPHA = 0.2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

profile_manager = ProfileManager()

cv2.resizeWindow("", CRT_WIDTH, CRT_HEIGHT)

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

    ghost = cv2.addWeighted(
        frame.astype(float),
        GHOST_ALPHA,
        ghost,
        1 - GHOST_ALPHA,
        0
    )
    display_frame = ghost.astype("uint8")

    # update profiles
    profiles = profile_manager.update(faces)
    for info in profiles.values():
        x, y, w, h = info["coords"]
        profile = info["profile"]

        # face box
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # profile panel
        name_line = f"{profile['name']} (Lvl {profile['level']})"
        stat_lines = [
            profile["class"],
            f"HP: {profile['health']}%",
            f"Trait: {profile['trait']}"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        padding = 6
        vertical_gap = 6

        # measure text
        line_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1]
        max_width = cv2.getTextSize(name_line, font, font_scale, thickness)[0][0]

        for line in stat_lines:
            w_line = cv2.getTextSize(line, font, font_scale, thickness)[0][0]
            max_width = max(max_width, w_line)

        rect_width = max_width + padding * 2
        rect_height = (
            line_height * (1 + len(stat_lines))
            + padding * 2
            + vertical_gap * 2
        )

        x_text = x + w + 8
        y_text = y

        # background panel (silver)
        cv2.rectangle(
            display_frame,
            (x_text - padding, y_text - padding),
            (x_text + rect_width - padding, y_text + rect_height - padding),
            (192, 192, 192),
            -1
        )

        # name
        name_y = y_text + line_height
        cv2.putText(
            display_frame,
            name_line,
            (x_text, name_y),
            font,
            font_scale,
            (0, 0, 0),
            thickness
        )

        # separator
        sep_y = name_y + vertical_gap
        cv2.line(
            display_frame,
            (x_text, sep_y),
            (x_text + rect_width - 2 * padding, sep_y),
            (0, 0, 0),
            1
        )

        # stats
        for i, line in enumerate(stat_lines):
            y_line = sep_y + vertical_gap + (i + 1) * line_height
            cv2.putText(
                display_frame,
                line,
                (x_text, y_line),
                font,
                font_scale,
                (0, 0, 0),
                thickness
            )

    # scanlines
    for y in range(0, display_frame.shape[0], LINE_SPACING):
        display_frame[y:y + 1] = (
            display_frame[y:y + 1] * LINE_INTENSITY
        ).astype("uint8")

    # CRT resize + show
    display_frame = cv2.resize(display_frame, (CRT_WIDTH, CRT_HEIGHT))
    cv2.imshow("", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
