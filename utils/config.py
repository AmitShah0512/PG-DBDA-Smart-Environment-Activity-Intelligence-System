import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "data", "processed", "har_video_dataset.npz")
MODEL_PATH = os.path.join(BASE_DIR, "model", "har_video_model.pth")

# Realtime parameters (tuned for webcam + phone-video demo)
SEQUENCE_LENGTH = 30          # keep original model window
CONF_THRESHOLD = 0.50         # was 0.70 → allow practical confidence
STABILITY_FRAMES = 3          # was 8 → allow faster DB logging

DB_PATH = os.path.join(BASE_DIR, "analytics", "actions_runtime.db")
EXPORT_CSV = os.path.join(BASE_DIR, "outputs", "actions_export.csv")
