import os
import uuid
from flask import Flask, render_template, request, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "fitai-preprocess-secret"
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 500  # 500MB


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            flash("No file uploaded.")
            return redirect(url_for("index"))

        file = request.files["video"]
        if file.filename == "":
            flash("Please choose a video file.")
            return redirect(url_for("index"))

        if not allowed_file(file.filename):
            flash("Unsupported format. Use mp4, mov, avi, mkv, webm.")
            return redirect(url_for("index"))

        safe_name = secure_filename(file.filename)
        token = uuid.uuid4().hex
        input_path = os.path.join(UPLOAD_DIR, f"{token}_{safe_name}")
        output_csv = os.path.join(OUTPUT_DIR, f"preprocessed_{token}.csv")

        file.save(input_path)

        try:
            from preprocess import preprocess_video_to_csv

            stats = preprocess_video_to_csv(input_path, output_csv)
        except Exception as exc:
            flash(f"Preprocessing failed: {exc}")
            return redirect(url_for("index"))
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

        return render_template(
            "result.html",
            csv_name=os.path.basename(output_csv),
            rows=stats.get("rows", 0),
            fps=stats.get("fps", 0),
            duration=stats.get("duration", 0),
        )

    return render_template("index.html")


@app.route("/download/<csv_name>")
def download(csv_name: str):
    safe_name = secure_filename(csv_name)
    file_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(file_path):
        flash("CSV file not found.")
        return redirect(url_for("index"))
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
