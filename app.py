from pathlib import Path
import os
import secrets
import logging

from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException
from PIL import Image, UnidentifiedImageError

# -----------------------------
# Flask App Configuration
# -----------------------------
app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB limit
app.config["UPLOAD_FOLDER"] = Path(app.root_path) / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app.logger = logger

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Global model cache
processor = None
model = None
device = None
torch = None


# -----------------------------
# Helper Functions
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load BLIP model only once."""
    global processor, model, device, torch

    if processor is not None and model is not None:
        return

    app.logger.info("Loading BLIP Image Captioning Model...")

    os.environ["USE_TF"] = "0"

    import torch as t
    from transformers import BlipProcessor, BlipForConditionalGeneration

    model_id = "Salesforce/blip-image-captioning-base"

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)

    device = "cuda" if t.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    torch = t

    app.logger.info(f"BLIP model loaded successfully on {device}")


def generate_caption(image):
    """Generate caption from uploaded image."""

    load_model()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=25
        )

    caption = processor.decode(output[0], skip_special_tokens=True).strip()

    if caption:
        caption = caption.capitalize()
        if not caption.endswith("."):
            caption += "."

    return caption


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
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

        app.logger.info(f"Caption Generated: {caption}")

        return jsonify({"caption": caption})

    except UnidentifiedImageError:
        return jsonify({"error": "Invalid image file."}), 400

    except Exception as e:
        app.logger.exception("Caption generation failed.")
        return jsonify({"error": str(e)}), 500

    finally:
        try:
            if image_path.exists():
                image_path.unlink()
        except Exception as e:
            app.logger.warning(f"Couldn't delete temp image: {e}")


# -----------------------------
# Error Handlers
# -----------------------------
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Page not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal Server Error."}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    if isinstance(e, HTTPException):
        return jsonify({"error": e.description}), e.code

    app.logger.exception("Unhandled exception")
    return jsonify({"error": str(e)}), 500


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)