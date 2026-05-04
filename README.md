# FitAI Preprocessing Tool

A Flask-based web app to upload workout videos and generate preprocessed CSV files using MediaPipe pose landmarks.

## Features
- Video upload from browser
- Pose landmark extraction (33 body landmarks/frame)
- Landmark normalization using torso length and hip center
- CSV download after preprocessing

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in browser.

## Output CSV
Each row is one frame with:
- `frame_idx`, `timestamp_sec`, `torso_length`
- For every landmark `i` in 0..32:
  - `lm_i_x`, `lm_i_y`, `lm_i_z`, `lm_i_vis`

## Notes
- If no person pose is detected in frames, preprocessing will fail with a clear error.
- Uploaded videos are deleted after processing; CSV outputs are stored in `outputs/`.
