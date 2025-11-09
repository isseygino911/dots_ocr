---
title: dots.ocr Parser
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
---

# DotsOCR Document Parser API

Multilingual document parsing powered by dots.ocr - 1.7B parameter Vision-Language Model for Document Understanding.

**Live API**: https://isseygino911-dots-ocr-parser.hf.space

## 🚀 Quick Links

- **[API Usage Guide](./API_USAGE_GUIDE.md)** - Complete examples for Python, JavaScript, React
- **[Interactive API Docs](https://isseygino911-dots-ocr-parser.hf.space/docs)** - Test endpoints
- **[Quick Start](./QUICKSTART.md)** - 5-minute setup guide
- **[Deployment](./DEPLOYMENT.md)** - Production deployment guide

## ⚡ Quick Test

```bash
# Test with Python
python test_upload.py your_document.pdf

# Or use cURL
curl -X POST https://isseygino911-dots-ocr-parser.hf.space/api/parse/image \
  -F "file=@image.jpg" \
  -F "prompt_mode=prompt_layout_all_en"
```

## 📖 What It Does

- **Extract text** from images and PDFs with high accuracy
- **Detect layout** - identifies titles, paragraphs, tables, figures, etc.
- **Generate structured output** - Markdown + JSON with bounding boxes
- **Process PDFs** - Handles multi-page documents automatically
- **Real-time progress** - WebSocket and HTTP polling support

## 🛠️ Features

- FastAPI backend with automatic OpenAPI documentation
- GPU-accelerated processing on HuggingFace Spaces
- RESTful API with WebSocket support for real-time updates
- Comprehensive examples for Python, JavaScript, and React
- Download results as ZIP with annotated images

## 📚 Documentation

See [API_USAGE_GUIDE.md](./API_USAGE_GUIDE.md) for:
- Complete API reference
- Python backend examples
- JavaScript/React frontend examples
- Error handling best practices
- WebSocket integration

## 🔧 Local Testing

```bash
# Clone the repo
git clone https://huggingface.co/spaces/isseygino911/dots-ocr-parser
cd dots-ocr-parser

# Test the API
python test_upload.py test.png
```

## 📄 License

Apache 2.0 - See dots.ocr repository for model license details