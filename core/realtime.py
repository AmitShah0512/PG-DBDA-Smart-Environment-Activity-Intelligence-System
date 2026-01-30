import cv2
import torch
import numpy as np
import time
from collections import deque, defaultdict

from pose.extractor import PoseExtractor
from model.net import CNNBiLSTM
from utils.config import MODEL_PATH, CONF_THRESHOLD, STABILITY_FRAMES
from db.logger import ActionLogger
import analytics.export as exporter


# --- Practical runtime tuning ---
WINDOW = 20              # Responsive window
WARMUP_SECONDS = 6        # Short warmup
DISPLAY_SMOOTH = 2        # Light UI smoothing

# Restrict to realistic room actions
ALLOWED_ACTIONS = {
    "PushUps",
    "JumpingJack",
    "BodyWeightSquats",
    "Lunges",
    "Punch",
    "WallPushups",
    "TaiChi",
    "PullUps",
}


def run_realtime():
    print("Starting Smart Environment Activity Intelligence System...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    if "class_names" in ckpt:
        class_names = ckpt["class_names"]
    else:
        inv = {v: k for k, v in ckpt["label_map"].items()}
        class_names = [inv[i] for i in range(len(inv))]

    model = CNNBiLSTM(len(class_names))
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    extractor = PoseExtractor()
    logger = ActionLogger()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    buf = deque(maxlen=WINDOW)
    stable = deque(maxlen=STABILITY_FRAMES)

    # UI helpers
    display_buf = deque(maxlen=DISPLAY_SMOOTH)
    last_display = "Collecting..."
    counts = defaultdict(int)

    start_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pose = extractor.extract(frame)
            if pose is not None:
                buf.append(pose)

            text = last_display
            conf = 0.0

            if len(buf) == WINDOW:
                seq = torch.tensor(np.array(buf), dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    out = model(seq)
                    probs = torch.softmax(out, 1)
                    conf_t, idx = probs.max(1)

                conf = float(conf_t.item())
                lbl = class_names[int(idx.item())]

                if lbl not in ALLOWED_ACTIONS:
                    lbl = None

                if time.time() - start_time < WARMUP_SECONDS:
                    text = "Warming up..."
                else:
                    if lbl is not None:
                        # Strict DB logging
                        if conf >= CONF_THRESHOLD:
                            stable.append(lbl)
                        else:
                            stable.clear()

                        if len(stable) == STABILITY_FRAMES and len(set(stable)) == 1:
                            logger.log(stable[0], conf)
                            counts[stable[0]] += 1
                            last_display = f"Logged: {stable[0]} ({conf:.2f})"
                            display_buf.clear()
                            stable.clear()

                        # UI smoothing (display only)
                        display_buf.append(lbl)
                        if len(display_buf) == display_buf.maxlen and len(set(display_buf)) == 1:
                            last_display = f"{lbl} ({conf:.2f})"

                    text = last_display

            # --- Overlays ---
            # System Title
            cv2.putText(frame, "Smart Environment Activity Intelligence System",
                        (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Main status
            cv2.putText(frame, text, (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Session timer
            elapsed = int(time.time() - start_time)
            mm, ss = elapsed // 60, elapsed % 60
            cv2.putText(frame, f"Time: {mm:02d}:{ss:02d}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            # Confidence bar
            bar_w = int(conf * 200)
            cv2.rectangle(frame, (20, 115), (20 + bar_w, 135), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 115), (220, 135), (255, 255, 255), 1)

            # Live counters (top 3)
            y = 170
            for k, v in list(counts.items())[:3]:
                cv2.putText(frame, f"{k}: {v}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                y += 25

            cv2.imshow("Smart Environment Activity Intelligence System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        extractor.release()
        cv2.destroyAllWindows()

        # Auto-export on exit (robust)
        try:
            if hasattr(exporter, "export_csv"):
                exporter.export_csv()
            elif hasattr(exporter, "export"):
                exporter.export()
            elif hasattr(exporter, "main"):
                exporter.main()
            else:
                print("No export function found in analytics.export")
            print("Session exported to CSV.")
        except Exception as e:
            print("Export failed:", e)
