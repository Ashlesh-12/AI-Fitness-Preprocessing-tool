import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

KP_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow",
    "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index",
    "left_thumb", "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
    "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]

EXERCISE_CANONICAL_MAP = {
    "abs": "standing_dumbbell_side_bend",
    "back": "bent_over_dumbbell_row",
    "bicep_curl": "alternating_dumbbell_bicep_curl",
    "push_up": "standard_push_up",
    "shoulder": "standing_dumbbell_shoulder_press",
    "squat": "bodyweight_squat",
    "tricep": "overhead_dumbbell_triceps_extension",
}

EXERCISE_GROUP_MAP = {
    "standing_dumbbell_side_bend": "abs_oblique",
    "bent_over_dumbbell_row": "back",
    "alternating_dumbbell_bicep_curl": "bicep",
    "standard_push_up": "push_chest_triceps",
    "standing_dumbbell_shoulder_press": "shoulder",
    "bodyweight_squat": "legs",
    "overhead_dumbbell_triceps_extension": "triceps",
}


@dataclass
class PreprocessConfig:
    subject_id: str
    exercise_folder: str
    form_quality: str
    angle_view: str
    clip_limit: float = 2.0
    tile_size: int = 8
    min_visibility: float = 0.5


def calculate_angle(point_a: dict, point_b: dict, point_c: dict) -> float:
    a = np.array([point_a["x"], point_a["y"]])
    b = np.array([point_b["x"], point_b["y"]])
    c = np.array([point_c["x"], point_c["y"]])
    ba = a - b
    bc = c - b
    mba = np.linalg.norm(ba)
    mbc = np.linalg.norm(bc)
    if mba == 0 or mbc == 0:
        return 0.0
    cos_angle = np.clip(np.dot(ba, bc) / (mba * mbc), -1.0, 1.0)
    return round(float(np.degrees(np.arccos(cos_angle))), 2)


def get_all_angles(keypoints: dict) -> dict:
    if keypoints is None:
        return {}
    kp = keypoints
    angles = {}
    if all(k in kp for k in ["left_shoulder", "left_elbow", "left_wrist"]):
        angles["left_elbow"] = calculate_angle(kp["left_shoulder"], kp["left_elbow"], kp["left_wrist"])
    if all(k in kp for k in ["right_shoulder", "right_elbow", "right_wrist"]):
        angles["right_elbow"] = calculate_angle(kp["right_shoulder"], kp["right_elbow"], kp["right_wrist"])
    if all(k in kp for k in ["left_elbow", "left_shoulder", "left_hip"]):
        angles["left_shoulder"] = calculate_angle(kp["left_elbow"], kp["left_shoulder"], kp["left_hip"])
    if all(k in kp for k in ["right_elbow", "right_shoulder", "right_hip"]):
        angles["right_shoulder"] = calculate_angle(kp["right_elbow"], kp["right_shoulder"], kp["right_hip"])
    if all(k in kp for k in ["left_hip", "left_knee", "left_ankle"]):
        angles["left_knee"] = calculate_angle(kp["left_hip"], kp["left_knee"], kp["left_ankle"])
    if all(k in kp for k in ["right_hip", "right_knee", "right_ankle"]):
        angles["right_knee"] = calculate_angle(kp["right_hip"], kp["right_knee"], kp["right_ankle"])
    if all(k in kp for k in ["left_shoulder", "left_hip", "left_knee"]):
        angles["left_hip"] = calculate_angle(kp["left_shoulder"], kp["left_hip"], kp["left_knee"])
    if all(k in kp for k in ["right_shoulder", "right_hip", "right_knee"]):
        angles["right_hip"] = calculate_angle(kp["right_shoulder"], kp["right_hip"], kp["right_knee"])
    return angles


def normalize_lighting(frame_bgr: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def landmarks_to_dict(landmarks, width: int, height: int) -> dict[str, dict[str, float]]:
    output = {}
    for i, lm in enumerate(landmarks.landmark):
        if i < len(KP_NAMES):
            output[KP_NAMES[i]] = {
                "x": lm.x * width,
                "y": lm.y * height,
                "z": lm.z,
                "visibility": lm.visibility,
            }
    return output


def bbox_from_landmarks(landmarks, width: int, height: int, margin: float = 0.2):
    xs, ys = [], []
    for lm in landmarks.landmark:
        x = lm.x * width
        y = lm.y * height
        if 0 <= x <= width and 0 <= y <= height:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return 0, 0, width, height
    min_x, max_x = max(0, int(min(xs))), min(width - 1, int(max(xs)))
    min_y, max_y = max(0, int(min(ys))), min(height - 1, int(max(ys)))
    bw, bh = max_x - min_x, max_y - min_y
    min_x = max(0, int(min_x - bw * margin))
    max_x = min(width - 1, int(max_x + bw * margin))
    min_y = max(0, int(min_y - bh * margin))
    max_y = min(height - 1, int(max_y + bh * margin))
    return min_x, min_y, max_x, max_y


def build_csv_header(angle_cols: list[str]) -> list[str]:
    return (
        [
            "frame_id", "timestamp_ms", "subject_id", "exercise_folder", "canonical_exercise", "movement_group",
            "form_quality", "angle_view", "avg_visibility", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        ]
        + angle_cols
        + [f"kp_{n}_x" for n in KP_NAMES]
        + [f"kp_{n}_y" for n in KP_NAMES]
        + [f"kp_{n}_vis" for n in KP_NAMES]
    )


def preprocess_video_to_csv(video_path: Path, output_csv: Path, output_report: Path, cfg: PreprocessConfig) -> dict:
    canonical_exercise = EXERCISE_CANONICAL_MAP.get(cfg.exercise_folder, cfg.exercise_folder)
    movement_group = EXERCISE_GROUP_MAP.get(canonical_exercise, "unknown")
    dummy_kp = {n: {"x": 0, "y": 0, "z": 0, "visibility": 1.0} for n in KP_NAMES}
    angle_cols = list(get_all_angles(dummy_kp).keys())
    header = build_csv_header(angle_cols)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Cannot open uploaded video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps <= 0:
        fps = 30.0

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=1, smooth_landmarks=True, min_detection_confidence=0.6, min_tracking_confidence=0.6)

    frame_id = pose_frames = kept_frames = low_vis_frames = overexposed_frames = 0
    bright_mean_values = []

    with open(output_csv, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(header)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            normalized = normalize_lighting(frame, cfg.clip_limit, cfg.tile_size)
            gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
            bright_mean = float(np.mean(gray))
            bright_mean_values.append(bright_mean)
            if bright_mean > 235:
                overexposed_frames += 1

            rgb = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if not result.pose_landmarks:
                frame_id += 1
                continue

            pose_frames += 1
            h, w = normalized.shape[:2]
            kp = landmarks_to_dict(result.pose_landmarks, w, h)
            angles = get_all_angles(kp)
            avg_vis = float(np.mean([kp[n]["visibility"] for n in KP_NAMES]))
            if avg_vis < cfg.min_visibility:
                low_vis_frames += 1
                frame_id += 1
                continue

            x1, y1, x2, y2 = bbox_from_landmarks(result.pose_landmarks, w, h, margin=0.2)
            ts_ms = int((frame_id / fps) * 1000.0)

            row = [
                frame_id,
                ts_ms,
                cfg.subject_id,
                cfg.exercise_folder,
                canonical_exercise,
                movement_group,
                cfg.form_quality,
                cfg.angle_view,
                round(avg_vis, 4),
                x1,
                y1,
                x2,
                y2,
            ]
            row.extend(round(angles.get(col, 0.0), 4) for col in angle_cols)
            row.extend(round(kp[n]["x"], 2) for n in KP_NAMES)
            row.extend(round(kp[n]["y"], 2) for n in KP_NAMES)
            row.extend(round(kp[n]["visibility"], 4) for n in KP_NAMES)
            writer.writerow(row)

            kept_frames += 1
            frame_id += 1

    cap.release()
    pose.close()

    total_frames = frame_id
    report = {
        "video": str(video_path),
        "subject_id": cfg.subject_id,
        "exercise_folder": cfg.exercise_folder,
        "canonical_exercise": canonical_exercise,
        "movement_group": movement_group,
        "form_quality": cfg.form_quality,
        "angle_view": cfg.angle_view,
        "total_frames": total_frames,
        "pose_detected_frames": pose_frames,
        "kept_frames": kept_frames,
        "dropped_low_visibility_frames": low_vis_frames,
        "pose_detect_ratio": round(pose_frames / total_frames, 4) if total_frames else 0.0,
        "kept_ratio": round(kept_frames / total_frames, 4) if total_frames else 0.0,
        "overexposed_ratio": round(overexposed_frames / total_frames, 4) if total_frames else 0.0,
        "brightness_mean": round(float(np.mean(bright_mean_values)), 4) if bright_mean_values else 0.0,
        "status": "ok",
        "csv_path": str(output_csv),
    }

    with open(output_report, "w", encoding="utf-8") as f_report:
        json.dump(report, f_report, indent=2)

    return report
