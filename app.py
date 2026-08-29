from pathlib import Path
import os
import secrets
import logging
import gc

from flask import Flask, render_template, request, jsonify
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

# Hugging Face cache
os.environ["HF_HOME"] = "/tmp/huggingface"

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = Path("/tmp/uploads")
app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

processor = None
model = None
tokenizer = None
torch = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    global processor, model, tokenizer, torch

    if model is None:
        app.logger.info("Loading lightweight image caption model...")

        import torch as t
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

        torch = t

        MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

        model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
        processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        model.to("cpu")
        model.eval()

        app.logger.info("Model loaded successfully.")

    return processor, tokenizer, model, torch


def generate_caption(image):
    processor, tokenizer, model, torch = load_model()

    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    with torch.inference_mode():
        output_ids = model.generate(
            pixel_values,
            max_length=20,
            num_beams=2
        )

    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    del pixel_values, output_ids
    gc.collect()

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