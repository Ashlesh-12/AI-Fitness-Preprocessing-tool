from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.preprocessing import PreprocessConfig, preprocess_video_to_csv

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = OUTPUT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, dict] = {}

app = FastAPI(title="FitAI Preprocessing Tool API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/preprocess")
async def preprocess(
    video: UploadFile = File(...),
    subject_id: str = Form("subject_001"),
    exercise_folder: str = Form("squat"),
    form_quality: str = Form("good"),
    angle_view: str = Form("front"),
    clip_limit: float = Form(2.0),
    tile_size: int = Form(8),
    min_visibility: float = Form(0.5),
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Video file is required.")

    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
        raise HTTPException(status_code=400, detail="Unsupported video type.")

    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = UPLOAD_DIR / f"{job_id}{ext}"
    with input_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    csv_name = f"{subject_id}_{exercise_folder}_{form_quality}_{angle_view}.csv"
    report_name = f"{subject_id}_{exercise_folder}_{form_quality}_{angle_view}.json"
    csv_path = job_dir / csv_name
    report_path = job_dir / report_name

    cfg = PreprocessConfig(
        subject_id=subject_id.strip() or "subject_001",
        exercise_folder=exercise_folder.strip() or "unknown",
        form_quality=form_quality.strip() or "unknown",
        angle_view=angle_view.strip() or "unknown",
        clip_limit=float(clip_limit),
        tile_size=int(tile_size),
        min_visibility=float(min_visibility),
    )

    try:
        report = preprocess_video_to_csv(input_path, csv_path, report_path, cfg)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}")

    JOBS[job_id] = {
        "csv_path": str(csv_path),
        "report_path": str(report_path),
        "csv_name": csv_name,
    }

    return {
        "job_id": job_id,
        "report": report,
        "download_url": f"/api/download/{job_id}/csv",
        "report_url": f"/api/download/{job_id}/report",
    }


@app.get("/api/download/{job_id}/csv")
def download_csv(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(path=job["csv_path"], filename=job["csv_name"], media_type="text/csv")


@app.get("/api/download/{job_id}/report")
def download_report(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(path=job["report_path"], filename=Path(job["report_path"]).name, media_type="application/json")
