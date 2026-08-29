from pathlib import Path
import os
import secrets
import logging
import requests

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

# Flask App
app = Flask(__name__)

# Configuration
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit
app.config["UPLOAD_FOLDER"] = Path("/tmp/uploads")
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Hugging Face Router API
API_URL = "https://router.huggingface.co/hf-inference/models/Salesforce/blip-image-captioning-base"


# ----------------------------
# Check allowed file extensions
# ----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ----------------------------
# Generate Caption
# ----------------------------
def generate_caption(image):

    token = os.environ.get("HF_TOKEN")

    if not token:
        raise Exception("HF_TOKEN environment variable not found.")

    headers = {
        "Authorization": f"Bearer {token}"
    }

    temp_path = "/tmp/temp.jpg"
    image.save(temp_path)

    with open(temp_path, "rb") as img:
        response = requests.post(
            API_URL,
            headers=headers,
            data=img,
            timeout=60
        )

    response.raise_for_status()

    result = response.json()

    # Extract caption
    if isinstance(result, list):
        caption = result[0]["generated_text"]
    else:
        caption = result["generated_text"]

    caption = caption.capitalize()

    if not caption.endswith("."):
        caption += "."

    return caption


# ----------------------------
# Home Page
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------
# Caption API
# ----------------------------
@app.route("/caption", methods=["POST"])
def caption():

    uploaded = request.files.get("image")

    if uploaded is None or uploaded.filename == "":
        return jsonify({"error": "Please upload an image."}), 400

    if not allowed_file(uploaded.filename):
        return jsonify({"error": "Only JPG, JPEG, PNG and WEBP images are allowed."}), 400

    filename = f"{secrets.token_hex(8)}.{uploaded.filename.rsplit('.',1)[1].lower()}"
    image_path = app.config["UPLOAD_FOLDER"] / filename

    uploaded.save(image_path)

    try:
        app.logger.info(f"Processing image: {uploaded.filename}")

        image = Image.open(image_path).convert("RGB")

        caption = generate_caption(image)

        return jsonify({"caption": caption})

    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file."}), 400

    except requests.exceptions.HTTPError as e:
        app.logger.exception("Hugging Face API error")
        return jsonify({"error": f"Hugging Face API Error: {e.response.text}"}), 500

    except Exception as e:
        app.logger.exception("Caption generation failed.")
        return jsonify({"error": str(e)}), 500

    finally:
        if image_path.exists():
            image_path.unlink()


# ----------------------------
# Global Error Handler
# ----------------------------
@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code

    app.logger.exception("Unexpected Error")
    return jsonify({"error": str(e)}), 500


# ----------------------------
# Run Flask
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)