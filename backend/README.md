# FitAI Preprocessing Tool - Backend

## Setup

```powershell
cd "D:\FitAI Preprocessing tool\backend"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```powershell
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Health check: `http://127.0.0.1:8000/health`
