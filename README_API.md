# DotsOCR API - React Frontend + FastAPI Backend

## Overview

This project provides a modern web interface for the DotsOCR document parser using:
- **Backend**: FastAPI (deployed on HuggingFace Spaces with GPU)
- **Frontend**: React + TypeScript (deployed on local Mac or Hostinger)

## Features

### Backend (FastAPI)
- RESTful API endpoints for image and PDF parsing
- WebSocket support for real-time progress updates
- Automatic API documentation (FastAPI/OpenAPI)
- CORS enabled for frontend access
- GPU-accelerated OCR processing

### Frontend (React)
- Drag-and-drop file upload
- Real-time progress tracking with WebSocket
- Results viewer with tabs:
  - Markdown rendered preview
  - Raw markdown text
  - JSON data with syntax highlighting
- Page navigation for PDF documents
- Download results as ZIP
- Responsive design

## Quick Start

### Backend (HuggingFace Spaces)

1. **Push to HF Spaces:**
```bash
git add app.py requirements.txt
git commit -m "FastAPI backend"
git push hf main
```

2. **Wait for deployment** (~10-15 minutes)

3. **Access API docs:**
```
https://isseygino911-dots-ocr-parser.hf.space/docs
```

### Frontend (Local Development)

1. **Navigate to frontend:**
```bash
cd frontend
```

2. **Install dependencies:**
```bash
npm install
```

3. **Configure backend URL:**
```bash
# Edit .env.development
VITE_API_BASE_URL=https://isseygino911-dots-ocr-parser.hf.space
```

4. **Run dev server:**
```bash
npm run dev
```

5. **Open browser:**
```
http://localhost:5173
```

### Frontend (Deploy to Hostinger)

1. **Build production:**
```bash
cd frontend
npm run build
```

2. **Upload `dist/` folder to Hostinger via FTP or File Manager**

3. **Access your domain:**
```
https://yourdomain.com
```

## Project Structure

```
dots-ocr-parser/
├── app.py                    # FastAPI backend
├── app_gradio_backup.py      # Original Gradio version (backup)
├── requirements.txt          # Python dependencies
├── DEPLOYMENT.md             # Detailed deployment guide
├── README_API.md             # This file
│
└── frontend/                 # React application
    ├── src/
    │   ├── api/
    │   │   └── client.ts     # API client functions
    │   ├── components/
    │   │   ├── FileUpload.tsx
    │   │   ├── JobProgress.tsx
    │   │   └── ResultsViewer.tsx
    │   ├── types/
    │   │   └── index.ts      # TypeScript types
    │   └── App.tsx           # Main app component
    ├── .env.development      # Dev environment config
    ├── .env.production       # Prod environment config
    └── package.json          # Node dependencies
```

## API Endpoints

### POST /api/parse/image
Upload and parse an image.

**Request:**
- `file`: Image file (multipart)
- `prompt_mode`: Parsing mode (form field)

**Response:**
```json
{
  "job_id": "abc-123-def",
  "status": "queued"
}
```

### POST /api/parse/pdf
Upload and parse a PDF.

**Response:** Same as image endpoint

### GET /api/jobs/{job_id}/status
Get job status and progress.

**Response:**
```json
{
  "job_id": "abc-123-def",
  "status": "processing",
  "progress_percent": 50.0,
  "current_page": 5,
  "total_pages": 10,
  "message": "Processing page 5/10..."
}
```

### GET /api/jobs/{job_id}/results
Get parsed results (when completed).

### GET /api/jobs/{job_id}/download
Download results as ZIP.

### WS /api/jobs/{job_id}/stream
WebSocket for real-time updates.

## Parsing Modes

- **prompt_layout_all_en** (Default): Full layout detection + text recognition
- **prompt_layout_only_en**: Layout detection only (no text)
- **prompt_ocr**: Text extraction only (markdown output)

## Development

### Backend Development

The backend is designed to run on HuggingFace Spaces with GPU. Local development on Mac is not supported due to CUDA requirement.

To test backend changes:
1. Push to HF Spaces
2. Wait for rebuild
3. Test via API docs at `/docs`

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run dev server (hot reload enabled)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### TypeScript Types

All API types are defined in `frontend/src/types/index.ts`. Update these if you modify the backend API.

## Troubleshooting

### CORS Errors
Update `allow_origins` in `app.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "http://localhost:5173"],
    ...
)
```

### WebSocket Connection Failed
The frontend automatically falls back to HTTP polling if WebSocket fails.

### Backend Not Responding
- Check if HF Space is sleeping (visit URL to wake)
- Check HF Spaces logs for errors
- Verify GPU is enabled in Space settings

### Frontend Build Errors
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

## Performance Notes

- **First run**: ~10 minutes (model download)
- **Subsequent runs**: ~30 seconds (model cached)
- **Image processing**: ~10-15 seconds per image
- **PDF processing**: ~10-15 seconds per page

## Tech Stack

### Backend
- Python 3.9+
- FastAPI 0.104+
- Uvicorn (ASGI server)
- PyTorch 2.4+
- Transformers 4.51+
- Flash Attention 2

### Frontend
- React 18
- TypeScript
- Vite (build tool)
- Axios (HTTP client)
- React Dropzone (file upload)
- React Markdown (markdown rendering)
- React Syntax Highlighter (JSON display)

## Credits

Powered by [dots.ocr](https://github.com/rednote-hilab/dots.ocr) - A 1.7B parameter Vision-Language Model for Document Understanding.

## License

See the dots.ocr repository for model license information.

## Next Steps

1. Deploy backend to HuggingFace Spaces
2. Test API endpoints at `/docs`
3. Run frontend locally and test integration
4. Deploy frontend to Hostinger
5. See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed instructions

## Support

For detailed deployment instructions, see [DEPLOYMENT.md](./DEPLOYMENT.md).

For API documentation, visit `/docs` on your deployed backend.
