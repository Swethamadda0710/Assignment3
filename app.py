from pathlib import Path
import secrets
import os
import logging

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = Path(app.root_path) / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app.logger = logger

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
_captioner = None
_model_error = None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_captioner():
    """Load the pretrained captioning checkpoint once, on the first request."""
    global _captioner, _model_error
    if _captioner is not None:
        return _captioner
    if _model_error is not None:
        raise RuntimeError(_model_error)

    try:
        app.logger.info("Loading BLIP model for image captioning...")
        os.environ["USE_TF"] = "0"
        from transformers import BlipForConditionalGeneration, BlipProcessor
        import torch

        # Use lite model for better memory efficiency on resource-constrained environments
        model_id = "Salesforce/blip-image-captioning-lite"
        app.logger.info(f"Downloading model from {model_id}...")
        processor = BlipProcessor.from_pretrained(model_id, use_fast=False)
        model = BlipForConditionalGeneration.from_pretrained(model_id, use_safetensors=False)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        app.logger.info(f"Using device: {device}")
        model.to(device)
        model.eval()
        
        app.logger.info("Model loaded successfully!")
        _captioner = (processor, model, device, torch)
        return _captioner
    except Exception as exc:
        _model_error = (
            "The pretrained captioning model could not be loaded. "
            "Run `pip install -r requirements.txt` and try again. "
            f"Details: {exc}"
        )
        app.logger.error(f"Model loading failed: {_model_error}")
        raise RuntimeError(_model_error) from exc


def generate_caption(image: Image.Image) -> str:
    """Generate caption for image with proper error handling"""
    try:
        processor, model, device, torch = get_captioner()
        app.logger.info(f"Generating caption for image of size {image.size}...")
        
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=30, num_beams=5)
        caption = processor.decode(output[0], skip_special_tokens=True).strip()
        
        # Ensure proper capitalization and punctuation
        if caption:
            caption = caption[0].upper() + caption[1:] if len(caption) > 1 else caption.upper()
            if not caption.endswith("."):
                caption += "."
        
        app.logger.info(f"Generated caption: {caption}")
        return caption
    except Exception as exc:
        app.logger.error(f"Caption generation error: {exc}", exc_info=True)
        raise


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/caption")
def caption():
    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        return jsonify(error="Please choose an image first."), 400
    if not allowed_file(uploaded.filename):
        return jsonify(error="Use a JPG, PNG, or WEBP image."), 400

    filename = f"{secrets.token_hex(12)}.{uploaded.filename.rsplit('.', 1)[1].lower()}"
    image_path = app.config["UPLOAD_FOLDER"] / filename
    uploaded.save(image_path)

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        result = generate_caption(image)
        return jsonify(caption=result)
    except UnidentifiedImageError:
        return jsonify(error="That file is not a readable image."), 400
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 503
    except Exception as exc:
        app.logger.exception("Caption generation failed")
        return jsonify(error=f"Caption generation failed: {str(exc)}"), 500
    finally:
        image_path.unlink(missing_ok=True)


@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Endpoint not found."), 404


@app.errorhandler(500)
def server_error(e):
    app.logger.exception("Unhandled server error")
    return jsonify(error="Internal server error. Please try again."), 500


@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions and return JSON"""
    if isinstance(e, HTTPException):
        return jsonify(error=e.description), e.code
    app.logger.exception(f"Unhandled exception: {e}")
    return jsonify(error="An unexpected error occurred. Please try again."), 500


if __name__ == "__main__":
    app.run(debug=True)
