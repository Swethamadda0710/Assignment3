from pathlib import Path
import os
import secrets
import logging
import gc

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = Path("/tmp/uploads")
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

captioner = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global captioner

    if captioner is None:
        app.logger.info("Loading lightweight image caption model...")

        from transformers import pipeline

        captioner = pipeline(
            "image-to-text",
            model="ydshieh/vit-gpt2-coco-en",
            device=-1  # CPU
        )

        app.logger.info("Model loaded successfully.")

    return captioner


def generate_caption(image):
    captioner = load_model()

    result = captioner(image)

    caption = result[0]["generated_text"].strip().capitalize()

    if not caption.endswith("."):
        caption += "."

    gc.collect()

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

    app.logger.exception("Unexpected error")
    return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)