import random
import math

names = ["Zyra", "Korr", "Velox", "Mira", "Axel", "Nix", "Luma"]
classes = ["Technomancer", "Shadow Operative", "Cyber Monk", "Net Runner", "Bio Hacker"]
traits = [
    "Collector of digital souls",
    "Allergic to sunlight",
    "Sees in infrared",
    "Loves old CRTs",
    "Glitch enthusiast"
]

FRAME_TIMEOUT = 60
MATCH_DISTANCE = 50


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
    x1, y1, w1, h1 = f1
    x2, y2, w2, h2 = f2
    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2
    return math.hypot(cx1 - cx2, cy1 - cy2)


class ProfileManager:
    def __init__(self):
        self.profiles = {}

    def update(self, detected_faces):
        for pid in self.profiles:
            self.profiles[pid]["seen"] = False

        for (x, y, w, h) in detected_faces:
            matched_id = None

            for pid, info in self.profiles.items():
                if distance((x, y, w, h), info["coords"]) < MATCH_DISTANCE:
                    matched_id = pid
                    break

            if matched_id is not None:
                info = self.profiles[matched_id]
                info["coords"] = (x, y, w, h)
                info["last_seen"] = 0
                info["seen"] = True
            else:
                pid = random.randint(100000, 999999)
                self.profiles[pid] = {
                    "profile": generate_profile(),
                    "coords": (x, y, w, h),
                    "last_seen": 0,
                    "seen": True
                }

        to_delete = []
        for pid, info in self.profiles.items():
            if not info["seen"]:
                info["last_seen"] += 1
                if info["last_seen"] > FRAME_TIMEOUT:
                    to_delete.append(pid)

        for pid in to_delete:
            del self.profiles[pid]

        return self.profiles
