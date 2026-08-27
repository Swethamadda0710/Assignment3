from pathlib import Path
import os
import secrets
import logging

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

# Hugging Face cache folder
os.environ["HF_HOME"] = "/tmp/huggingface"

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = Path("uploads")
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

processor = None
model = None
torch = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global processor, model, torch

    if processor is None:
        app.logger.info("Loading BLIP caption model...")

        import torch as t
        from transformers import BlipProcessor, BlipForConditionalGeneration

        torch = t

        MODEL_NAME = "Salesforce/blip-image-captioning-base"

        processor = BlipProcessor.from_pretrained(MODEL_NAME)

        model = BlipForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True
        )

        model.to("cpu")
        model.eval()

        app.logger.info("BLIP model loaded successfully.")

    return processor, model, torch


def generate_caption(image):
    processor, model, torch = load_model()

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=15,
            num_beams=1,
            do_sample=False
        )

    caption = processor.decode(output[0], skip_special_tokens=True).strip()

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

    app.logger.exception("Unexpected error")
    return jsonify({"error": str(e)}), 500


# Load model when Render starts
try:
    load_model()
except Exception as e:
    app.logger.error(f"Startup model loading failed: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)