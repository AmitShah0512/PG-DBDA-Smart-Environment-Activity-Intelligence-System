import cv2
import numpy as np
import mediapipe as mp

class PoseExtractor:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)

        if not res.pose_landmarks:
            return None

        pts = []
        for lm in res.pose_landmarks.landmark:
            pts.append([lm.x, lm.y])

        return np.array(pts, dtype=np.float32)

    def release(self):
        self.pose.close()
