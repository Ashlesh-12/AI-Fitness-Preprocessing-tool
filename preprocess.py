import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


mp_pose = mp.solutions.pose



def _compute_distance(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))



def preprocess_video_to_csv(video_path: str, output_csv_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Unable to read the uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (frame_count / fps) if fps > 0 else 0

    rows = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                l_shoulder = landmarks[11]
                r_shoulder = landmarks[12]
                l_hip = landmarks[23]
                r_hip = landmarks[24]

                shoulder_center = (
                    (l_shoulder.x + r_shoulder.x) / 2,
                    (l_shoulder.y + r_shoulder.y) / 2,
                )
                hip_center = (
                    (l_hip.x + r_hip.x) / 2,
                    (l_hip.y + r_hip.y) / 2,
                )
                torso_length = _compute_distance(shoulder_center, hip_center)
                if torso_length < 1e-6:
                    torso_length = 1e-6

                frame_data = {
                    "frame_idx": frame_idx,
                    "timestamp_sec": frame_idx / fps if fps > 0 else 0,
                    "torso_length": torso_length,
                }

                for i, lm in enumerate(landmarks):
                    nx = (lm.x - hip_center[0]) / torso_length
                    ny = (lm.y - hip_center[1]) / torso_length
                    nz = lm.z / torso_length
                    frame_data[f"lm_{i}_x"] = nx
                    frame_data[f"lm_{i}_y"] = ny
                    frame_data[f"lm_{i}_z"] = nz
                    frame_data[f"lm_{i}_vis"] = lm.visibility

                rows.append(frame_data)

            frame_idx += 1

    cap.release()

    if not rows:
        raise ValueError("No detectable human pose landmarks found in video.")

    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)

    return {
        "rows": len(df),
        "fps": round(float(fps), 2),
        "duration": round(float(duration), 2),
    }
