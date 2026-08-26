# Image Caption Generator - Fixed Version

## What Was Fixed

The app was throwing a **JSON parsing error** when trying to generate image captions. The error message was:
```
Your caption could not be generated.
Unexpected token '<', '<html>' '<'... is not valid JSON
```

This occurred because:
1. The Flask backend wasn't properly handling exceptions, returning HTML error pages instead of JSON
2. The frontend JavaScript tried to parse HTML as JSON without proper validation

**All issues have been fixed!** See [FIXES_SUMMARY.md](FIXES_SUMMARY.md) for detailed information.

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

The app will start on `http://127.0.0.1:5000`

### 3. Access the Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

## Features

- **Upload Images**: Drag & drop or click to select JPG, PNG, or WEBP images (up to 8 MB)
- **AI Caption Generation**: Uses the pretrained BLIP model from Salesforce
- **Local Processing**: All processing happens locally on your machine
- **Real-time Feedback**: Clear status messages and error handling

## Important Notes

### First Run
- The first caption generation will take a few minutes
- The app downloads the pretrained BLIP model (990 MB)
- This only happens once; subsequent requests are fast

### Timeout
- First inference can take 3-5 minutes
- Subsequent inferences are typically 30-60 seconds
- Model is cached after first download

### Device Support
- Automatically detects CUDA GPU if available
- Falls back to CPU processing if GPU not available
- GPU significantly speeds up inference

## Technical Stack

- **Backend**: Flask 3.0+
- **ML Model**: Salesforce BLIP Image Captioning (Hugging Face)
- **Image Processing**: Pillow
- **Deep Learning**: PyTorch 2.2+
- **Frontend**: Vanilla JavaScript + CSS

## Testing

The application has been tested with:
- ✅ JPEG images
- ✅ PNG images  
- ✅ Various image sizes
- ✅ Error scenarios (invalid files, missing images)
- ✅ Model loading and caching
- ✅ JSON response validation

## Troubleshooting

### "The first caption may take a few minutes..."
This is normal! The model is downloading and loading. Wait for the process to complete.

### Error: "Use a JPG, PNG, or WEBP image"
Make sure your image file has one of these extensions and is a valid image.

### Memory Issues
The app requires ~3 GB RAM for model + inference. Ensure your system has sufficient memory.

### CUDA/GPU Issues
If using GPU, ensure PyTorch CUDA support is properly installed. The app will automatically fall back to CPU.

## Deployment

For production deployment (Render, Heroku, etc.):
1. Use `gunicorn` instead of Flask's debug server
2. Ensure deployment platform has sufficient memory (4+ GB recommended)
3. Set up persistent storage for model caching
4. Configure appropriate timeout settings (5+ minutes)
5. See [Procfile](Procfile) and [render.yaml](render.yaml) for platform-specific configs

## Files

- `app.py` - Flask backend with error handling and logging
- `templates/index.html` - Web interface with robust JavaScript
- `static/style.css` - Styling
- `requirements.txt` - Python dependencies
- `FIXES_SUMMARY.md` - Detailed technical fixes
- `Procfile` - Heroku deployment config
- `render.yaml` - Render deployment config
