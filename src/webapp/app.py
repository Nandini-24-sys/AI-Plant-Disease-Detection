import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from ml.infer import predict_with_gradcam

# -----------------------
# Flask App Setup
# -----------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pth")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------
# Home Page
# -----------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------
# Upload Page
# -----------------------
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        # Check file exists
        if "image" not in request.files:
            return "No file uploaded!"

        file = request.files["image"]

        if file.filename == "":
            return "No selected file!"

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # Predict
        prediction, confidence, gradcam_path = predict_with_gradcam(save_path)

        return render_template(
            "result.html",
            filename=filename,
            prediction=prediction,
            confidence=confidence,
            gradcam_path=gradcam_path
        )

    return render_template("upload.html")


# -----------------------
# Serve Image Uploads
# -----------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# -----------------------
# Serve Grad-CAM files
# -----------------------
@app.route("/gradcam/<filename>")
def gradcam_file(filename):
    gradcam_dir = os.path.join(BASE_DIR, "gradcam_outputs")
    return send_from_directory(gradcam_dir, filename)


# -----------------------
# Start App
# -----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
