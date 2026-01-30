import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import time
import torch
import numpy as np
from collections import deque, defaultdict

from pose.extractor import PoseExtractor
from model.net import CNNBiLSTM
from utils.config import MODEL_PATH, CONF_THRESHOLD, STABILITY_FRAMES
from db.logger import ActionLogger

TITLE_FONT = ("Bahnschrift", 26, "bold")
SUBTITLE_FONT = ("Consolas", 13)
PANEL_TITLE_FONT = ("Bahnschrift", 14, "bold")
STAT_NAME_FONT = ("Consolas", 12)
STAT_VALUE_FONT = ("Bahnschrift", 12, "bold")
COUNTS_TITLE_FONT = ("Bahnschrift", 12, "bold")
COUNTS_VALUE_FONT = ("Consolas", 11)
BUTTON_FONT = ("Bahnschrift", 14, "bold")

# ---- Realtime tuning (balanced) ----
WINDOW = 12              # was 20 → react faster
DISPLAY_SMOOTH = 2
WARMUP_SECONDS = 3       # was 6 → start sooner
FRAME_DELAY_MS = 15
COOLDOWN_SEC = 0.8       # prevents burst increments

ALLOWED_ACTIONS = {
    "PushUps", "JumpingJack", "BodyWeightSquats", "Lunges",
    "Punch", "WallPushups", "TaiChi", "PullUps",
}

class SmartEnvUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Environment Activity Intelligence System")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#0b0f14")

        self.running = False
        self.cap = None
        self.last_log_time = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)

        if "class_names" in ckpt:
            self.class_names = ckpt["class_names"]
        else:
            inv = {v: k for k, v in ckpt["label_map"].items()}
            self.class_names = [inv[i] for i in range(len(inv))]

        self.model = CNNBiLSTM(len(self.class_names))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()

        self.extractor = PoseExtractor()
        self.logger = ActionLogger()

        self.buf = deque(maxlen=WINDOW)
        self.stable = deque(maxlen=STABILITY_FRAMES)
        self.display_buf = deque(maxlen=DISPLAY_SMOOTH)
        self.last_display = "Idle"
        self.counts = defaultdict(int)

        header = tk.Frame(root, bg="#0b0f14")
        header.pack(fill="x", pady=10)

        tk.Label(header, text="SMART ENVIRONMENT ACTIVITY INTELLIGENCE SYSTEM",
                 fg="#00ffd5", bg="#0b0f14", font=TITLE_FONT).pack()

        tk.Label(header, text="Real-Time Human Behavior Monitoring & Analytics",
                 fg="#6ee7ff", bg="#0b0f14", font=SUBTITLE_FONT).pack()

        main = tk.Frame(root, bg="#0b0f14")
        main.pack(fill="both", expand=True, padx=20, pady=20)
        main.columnconfigure(0, weight=3)
        main.columnconfigure(1, weight=2)

        cam_frame = tk.Frame(main, bg="#111827", highlightbackground="#00ffd5", highlightthickness=2)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))

        tk.Label(cam_frame, text="LIVE FEED", fg="#00ffd5",
                 bg="#111827", font=PANEL_TITLE_FONT).pack(anchor="nw", padx=10, pady=10)

        self.cam_label = tk.Label(cam_frame, bg="#111827")
        self.cam_label.pack(expand=True, fill="both")

        status_frame = tk.Frame(main, bg="#111827", highlightbackground="#6ee7ff", highlightthickness=2)
        status_frame.grid(row=0, column=1, sticky="nsew")

        tk.Label(status_frame, text="SYSTEM STATUS", fg="#6ee7ff",
                 bg="#111827", font=PANEL_TITLE_FONT).pack(anchor="nw", padx=10, pady=10)

        self.lbl_action = self._stat(status_frame, "Action", "Idle")
        self.lbl_conf = self._stat(status_frame, "Confidence", "--")
        self.lbl_time = self._stat(status_frame, "Session", "00:00")

        self.counts_box = tk.Frame(status_frame, bg="#111827")
        self.counts_box.pack(fill="x", padx=10, pady=20)

        tk.Label(self.counts_box, text="ACTIVITY COUNTS",
                 fg="#00ffd5", bg="#111827", font=COUNTS_TITLE_FONT).pack(anchor="w")

        self.count_labels = {}

        controls = tk.Frame(root, bg="#0b0f14")
        controls.pack(fill="x", pady=10)

        btn_style = dict(font=BUTTON_FONT, fg="#0b0f14", bg="#00ffd5",
                         activebackground="#6ee7ff", bd=0, padx=20, pady=10)

        tk.Button(controls, text="START SESSION", **btn_style, command=self.start).pack(side="left", padx=20)
        tk.Button(controls, text="STOP", **btn_style, command=self.stop).pack(side="left", padx=20)
        tk.Button(controls, text="EXIT", font=BUTTON_FONT, fg="#fff",
                  bg="#ff1744", activebackground="#ff616f", bd=0,
                  padx=20, pady=10, command=self.on_exit).pack(side="right", padx=20)

    def _stat(self, parent, name, value):
        f = tk.Frame(parent, bg="#111827")
        f.pack(fill="x", padx=10, pady=5)
        tk.Label(f, text=f"{name} :", fg="#9ca3af",
                 bg="#111827", font=STAT_NAME_FONT).pack(side="left")
        lbl = tk.Label(f, text=value, fg="#00ffd5",
                       bg="#111827", font=STAT_VALUE_FONT)
        lbl.pack(side="right")
        return lbl

    def start(self):
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.cap = cv2.VideoCapture(0)
        self.update_loop()

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def update_loop(self):
        if not self.running or not self.cap:
            return

        ok, frame = self.cap.read()
        if ok:
            pose = self.extractor.extract(frame)
            if pose is not None:
                self.buf.append(pose)

            conf = 0.0
            text = self.last_display

            if len(self.buf) == WINDOW:
                seq = torch.tensor(np.array(self.buf), dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(seq)
                    probs = torch.softmax(out, 1)
                    conf_t, idx = probs.max(1)

                conf = float(conf_t.item())
                lbl = self.class_names[int(idx.item())]

                if lbl not in ALLOWED_ACTIONS:
                    lbl = None

                if time.time() - self.start_time > WARMUP_SECONDS and lbl:
                    if conf >= CONF_THRESHOLD:
                        self.stable.append(lbl)
                    else:
                        self.stable.clear()

                    now = time.time()
                    if len(self.stable) == STABILITY_FRAMES and len(set(self.stable)) == 1:
                        if now - self.last_log_time >= COOLDOWN_SEC:
                            self.logger.log(lbl, conf)
                            self.counts[lbl] += 1
                            self.last_log_time = now
                        self.stable.clear()

                    self.display_buf.append(lbl)
                    if len(self.display_buf) == self.display_buf.maxlen and len(set(self.display_buf)) == 1:
                        self.last_display = lbl

                    text = self.last_display

            self.lbl_action.config(text=text)
            self.lbl_conf.config(text=f"{conf:.2f}")
            elapsed = int(time.time() - self.start_time)
            self.lbl_time.config(text=f"{elapsed//60:02d}:{elapsed%60:02d}")

            for k, v in self.counts.items():
                if k not in self.count_labels:
                    lbl = tk.Label(self.counts_box, text="", fg="#6ee7ff",
                                   bg="#111827", font=COUNTS_VALUE_FONT)
                    lbl.pack(anchor="w")
                    self.count_labels[k] = lbl
                self.count_labels[k].config(text=f"{k} : {v}")

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame).resize((900, 600))
            imgtk = ImageTk.PhotoImage(image=img)
            self.cam_label.imgtk = imgtk
            self.cam_label.configure(image=imgtk)

        self.root.after(FRAME_DELAY_MS, self.update_loop)

    def on_exit(self):
        self.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartEnvUI(root)
    root.mainloop()
