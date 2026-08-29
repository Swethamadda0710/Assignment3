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

# Hugging Face Model API
API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_caption(image):
    """
    Sends image to Hugging Face Inference API and returns caption.
    """

    token = os.getenv("HF_TOKEN")

    if not token:
        raise Exception("HF_TOKEN environment variable not found.")

    temp_path = "/tmp/temp_image.jpg"
    image.save(temp_path)

    headers = {
        "Authorization": f"Bearer {token}"
    }

    with open(temp_path, "rb") as img:
        response = requests.post(API_URL, headers=headers, data=img)

    if response.status_code != 200:
        raise Exception(response.text)

    result = response.json()

    if isinstance(result, list):
        caption = result[0]["generated_text"]
    else:
        caption = result["generated_text"]

    caption = caption.capitalize()

    if not caption.endswith("."):
        caption += "."

    return caption


@app.route("/")
def index():
    return render_template("index.html")


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

    except Exception as e:
        app.logger.exception("Caption generation failed.")
        return jsonify({"error": str(e)}), 500

    finally:
        if image_path.exists():
            image_path.unlink()


@app.errorhandler(Exception)
def handle_exception(e):

    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code

    app.logger.exception("Unexpected Error")
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)