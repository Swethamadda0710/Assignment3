---
title: Frame Caption
emoji: "🖼️"
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

# Frame / Caption

A Flask image-captioning application using a downloaded pretrained vision-language checkpoint. Upload a JPG, PNG, or WEBP image and receive a natural-language caption.

## Run locally

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000. The first caption request downloads `Salesforce/blip-image-captioning-base` from Hugging Face and caches it locally. No training is performed. CPU inference works, but a CUDA-enabled PyTorch install is faster.

## Architecture note

The requested CNN Encoder + LSTM Decoder pattern is the classic image-captioning architecture. This implementation uses a current, pretrained encoder-decoder checkpoint so it can generate useful captions without training or shipping a separate custom vocabulary. BLIP's released checkpoint uses a vision encoder and transformer text decoder rather than an LSTM; replacing it with a CNN+LSTM checkpoint requires a compatible pretrained decoder, vocabulary, and checkpoint files.
