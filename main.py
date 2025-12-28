import cv2
import numpy as np
import random

CRT_WIDTH, CRT_HEIGHT = 720, 480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

ghost = None
LINE_SPACING = 4        # scanline spacing
LINE_INTENSITY = 0.9    # 0 = black line, 1 = unchanged

names = ["Zyra", "Korr", "Velox", "Mira", "Axel", "Nix", "Luma"]
classes = ["Technomancer", "Shadow Operative", "Cyber Monk", "Net Runner", "Bio Hacker"]
traits = ["Collector of digital souls", "Allergic to sunlight", "Sees in infrared",
          "Loves old CRTs", "Glitch enthusiast"]

# Profiles with timeout tracking
face_profiles = {}
FRAME_TIMEOUT = 60  # frames to keep profile after face disappears

def generate_profile():
    return {
        "name": random.choice(names),
        "level": random.randint(1, 99),
        "health": random.randint(50, 100),
        "strength": random.randint(1, 10),
        "agility": random.randint(1, 10),
        "intellect": random.randint(1, 10),
        "class": random.choice(classes),
        "trait": random.choice(traits)
    }

def distance(f1, f2):
    # Euclidean distance between face centers
    x1, y1, w1, h1 = f1
    x2, y2, w2, h2 = f2
    cx1, cy1 = x1 + w1/2, y1 + h1/2
    cx2, cy2 = x2 + w2/2, y2 + h2/2
    return ((cx1 - cx2)**2 + (cy1 - cy2)**2)**0.5

cv2.namedWindow("CRT FACE", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CRT FACE", CRT_WIDTH, CRT_HEIGHT)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if ghost is None:
        ghost = frame.astype(float)

    # Ghosting
    alpha = 0.4
    ghost = cv2.addWeighted(frame.astype(float), alpha, ghost, 1 - alpha, 0)
    display_frame = ghost.astype('uint8')

    # Mark all profiles as unseen
    for info in face_profiles.values():
        info['seen_this_frame'] = False

    for (x, y, w, h) in faces:
        # Match face to existing profiles
        matched_id = None
        for pid, info in face_profiles.items():
            if distance((x, y, w, h), info['coords']) < 50:
                matched_id = pid
                break

        if matched_id is not None:
            profile_info = face_profiles[matched_id]
            profile_info['coords'] = (x, y, w, h)
            profile_info['last_seen'] = 0
            profile_info['seen_this_frame'] = True
            profile = profile_info['profile']
        else:
            profile = generate_profile()
            new_id = random.randint(100000, 999999)
            face_profiles[new_id] = {
                'profile': profile,
                'coords': (x, y, w, h),
                'last_seen': 0,
                'seen_this_frame': True
            }

        # Draw face rectangle
        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

        # Profile text: name + separator + stats
        name_line = f"ID: {profile['name']} [lvl {profile['level']}]"
        other_lines = [
            f"HP: {profile['health']}%",
            f"Class: {profile['class']}",
            f"Trait: {profile['trait']}"
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        padding = 4

        # Determine max width dynamically
        max_width = cv2.getTextSize(name_line, font, font_scale, thickness)[0][0]
        line_height = cv2.getTextSize("Test", font, font_scale, thickness)[0][1]
        for line in other_lines:
            w_line, _ = cv2.getTextSize(line, font, font_scale, thickness)[0]
            if w_line > max_width:
                max_width = w_line

        rect_width = max_width + padding * 2
        rect_height = line_height * (1 + len(other_lines)) + padding * 3  # extra for separator

        x_text = x + w + 5
        y_text = y
        separator_padding = 6

        # Draw silver background
        cv2.rectangle(display_frame,
                      (x_text - padding, y_text - padding),
                      (x_text + rect_width - padding, y_text + rect_height - padding + separator_padding * 2),
                      (192, 192, 192), -1)

        # Draw name
        cv2.putText(display_frame, name_line, (x_text, y_text + line_height),
                    font, font_scale, (0,0,0), thickness)

        # Draw horizontal separator
        
        sep_y = y_text + line_height + separator_padding
        cv2.line(display_frame, (x_text, sep_y), (x_text + rect_width - padding, sep_y), (0,0,0), 1)

        # Draw other lines
        for i, line in enumerate(other_lines):
            y_line = sep_y + line_height + i*line_height + i*2 + separator_padding
            cv2.putText(display_frame, line, (x_text, y_line),
                        font, font_scale, (0,0,0), thickness)

    # Update profiles not seen
    to_delete = []
    for pid, info in face_profiles.items():
        if not info['seen_this_frame']:
            info['last_seen'] += 1
            if info['last_seen'] > FRAME_TIMEOUT:
                to_delete.append(pid)
    for pid in to_delete:
        del face_profiles[pid]

    # Scanlines
    for y in range(0, display_frame.shape[0], LINE_SPACING):
        display_frame[y:y+1, :, :] = (display_frame[y:y+1, :, :] * LINE_INTENSITY).astype('uint8')

    # Resize to CRT
    display_frame = cv2.resize(display_frame, (CRT_WIDTH, CRT_HEIGHT))
    cv2.imshow("CRT FACE", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
