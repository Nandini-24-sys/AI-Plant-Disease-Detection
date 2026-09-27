import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png","jpg","jpeg"}
DB_PATH = os.path.join(BASE_DIR, "app.db")
SECRET_KEY = "replace-with-a-secure-key"
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pth")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
