from pathlib import Path
import os
import secrets
import logging

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from huggingface_hub import InferenceClient

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = Path("/tmp/uploads")
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

client = InferenceClient(token=os.environ.get("HF_TOKEN"))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_caption(image):
    temp_path = "/tmp/image.jpg"
    image.save(temp_path)

    result = client.image_to_text(
        temp_path,
        model="Salesforce/blip-image-captioning-large"
    )

    caption = result.generated_text.capitalize()
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
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)