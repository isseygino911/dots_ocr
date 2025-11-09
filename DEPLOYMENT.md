# DotsOCR API - Deployment Guide

## Architecture Overview

This project consists of two main components:

1. **Backend (FastAPI)** - Runs on HuggingFace Spaces (requires GPU)
2. **Frontend (React)** - Runs on your local Mac or Hostinger

```
┌─────────────────────────────────────────────────┐
│  React Frontend (Mac local dev / Hostinger)     │
│  - File upload UI                               │
│  - Progress tracking                            │
│  - Results viewer                               │
└──────────────────┬──────────────────────────────┘
                   │ HTTPS API calls
                   ▼
┌─────────────────────────────────────────────────┐
│  FastAPI Backend (HuggingFace Spaces)           │
│  - REST API endpoints                           │
│  - WebSocket for progress                       │
│  - DotsOCRParser (GPU/CUDA)                     │
└─────────────────────────────────────────────────┘
```

---

## Part 1: Deploy Backend to HuggingFace Spaces

### Prerequisites
- HuggingFace account
- Git installed
- Your HuggingFace Spaces repository

### Step 1: Prepare Backend Files

The following files need to be in your HF Spaces repository:

```
your-hf-space/
├── app.py              # FastAPI backend (already modified)
├── requirements.txt    # Updated with FastAPI deps
├── .gitignore
└── README.md
```

### Step 2: Update .gitignore

Make sure your `.gitignore` includes:

```
# Data directories
data/
uploads/
results/
*.pyc
__pycache__/
.DS_Store

# Model weights (will be downloaded on startup)
dots.ocr/weights/

# Environment
.env
```

### Step 3: Push to HuggingFace Spaces

```bash
cd /Users/mac/Desktop/dots-ocr-parser

# Add HF Spaces as remote (if not already added)
git remote add hf https://huggingface.co/spaces/isseygino911/dots-ocr-parser

# Commit the changes
git add app.py requirements.txt
git commit -m "Replace Gradio with FastAPI backend"

# Push to HF Spaces
git push hf main
```

### Step 4: Configure HF Spaces Settings

1. Go to your Space settings on HuggingFace
2. Make sure **GPU** is enabled (required for CUDA)
3. Set **SDK** to `docker` or `gradio` (it will work with either)
4. The space will automatically rebuild (takes ~10-15 minutes)

### Step 5: Verify Backend is Running

Once deployed, visit:
- **API Root**: `https://isseygino911-dots-ocr-parser.hf.space/`
- **API Docs**: `https://isseygino911-dots-ocr-parser.hf.space/docs`
- **Health Check**: `https://isseygino911-dots-ocr-parser.hf.space/health`

You should see the FastAPI automatic documentation at `/docs`.

### Step 6: Test API Endpoints

Using curl or Postman, test the health endpoint:

```bash
curl https://isseygino911-dots-ocr-parser.hf.space/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu_available": true
}
```

---

## Part 2: Setup React Frontend (Local Development)

### Prerequisites
- Node.js 18+ and npm installed
- Backend deployed and running on HF Spaces

### Step 1: Navigate to Frontend Directory

```bash
cd /Users/mac/Desktop/dots-ocr-parser/frontend
```

### Step 2: Update Environment Variables

Edit `.env.development` to point to your HF Spaces backend:

```bash
# .env.development
VITE_API_BASE_URL=https://isseygino911-dots-ocr-parser.hf.space
```

### Step 3: Install Dependencies (if not already done)

```bash
npm install
```

### Step 4: Run Development Server

```bash
npm run dev
```

The app will be available at: `http://localhost:5173`

### Step 5: Test the Application

1. Open `http://localhost:5173` in your browser
2. Upload an image or PDF file
3. Watch the progress in real-time
4. View the results with markdown/JSON tabs
5. Download the results as ZIP

---

## Part 3: Deploy Frontend to Hostinger

### Prerequisites
- Hostinger account with hosting plan
- FTP access or File Manager access

### Step 1: Build Production Frontend

```bash
cd /Users/mac/Desktop/dots-ocr-parser/frontend

# Update production environment variable
echo "VITE_API_BASE_URL=https://isseygino911-dots-ocr-parser.hf.space" > .env.production

# Build for production
npm run build
```

This creates a `dist/` folder with static files.

### Step 2: Upload to Hostinger

**Option A: Using FTP**

1. Connect to Hostinger via FTP (use FileZilla or similar)
2. Navigate to your public_html folder (or subdirectory)
3. Upload all files from `frontend/dist/` to the server
4. Make sure `index.html` is in the root of your domain folder

**Option B: Using File Manager**

1. Login to Hostinger control panel
2. Go to File Manager
3. Navigate to public_html
4. Upload the `dist.zip` (compress dist/ folder first)
5. Extract on server

### Step 3: Configure Hostinger

If using a subdirectory (e.g., `yourdomain.com/ocr`), update `vite.config.ts`:

```typescript
export default defineConfig({
  base: '/ocr/',  // Add this line
  plugins: [react()],
})
```

Then rebuild and re-upload.

### Step 4: Verify Deployment

Visit your domain:
- `https://yourdomain.com` (or subdirectory)
- Try uploading a file and check if it connects to HF backend

### Step 5: Update CORS on Backend (if needed)

If you get CORS errors, update `app.py` on HF Spaces:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Or specifically:
        "https://yourdomain.com",
        "http://localhost:5173"  # For dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Then push to HF Spaces again.

---

## Part 4: Troubleshooting

### Backend Issues

**Issue**: Model download takes too long
- **Solution**: HF Spaces has persistent storage. First run takes ~10 min, subsequent starts are faster.

**Issue**: CUDA out of memory
- **Solution**: Reduce `max_pixels` in `DEFAULT_CONFIG` or enable smaller model.

**Issue**: API returning 503
- **Solution**: Space may be sleeping. Visit the Space URL to wake it up.

### Frontend Issues

**Issue**: CORS errors in browser console
- **Solution**: Update `allow_origins` in backend CORS middleware.

**Issue**: WebSocket connection fails
- **Solution**: Falls back to polling automatically. Check browser console for details.

**Issue**: Results not loading
- **Solution**: Check Network tab in browser DevTools. Verify backend URL is correct.

### Common Problems

**Problem**: "Job not found" error
- **Cause**: Backend restarted and lost in-memory job data
- **Solution**: Use persistent storage (SQLite) for job metadata (optional enhancement)

**Problem**: Large PDFs timeout
- **Cause**: Processing takes > 5 minutes
- **Solution**: This is normal. WebSocket will keep updating progress.

---

## Part 5: Optional Enhancements

### Add SQLite for Job Persistence

Currently jobs are stored in-memory. To persist across restarts:

1. Install SQLite dependency:
```bash
pip install sqlalchemy aiosqlite
```

2. Add database models and persistence in `app.py`

### Add Authentication

To secure the API:

1. Add JWT authentication
2. Create user registration/login endpoints
3. Require API key for all parse requests

### Add Rate Limiting

To prevent abuse:

```bash
pip install slowapi
```

Add rate limiting middleware to FastAPI.

### Monitor with Logging

Add structured logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

---

## Part 6: API Reference

### Endpoints

#### POST /api/parse/image
Upload and parse an image file.

**Request:**
- `file` (multipart): Image file
- `prompt_mode` (form): Parsing mode (default: `prompt_layout_all_en`)

**Response:**
```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

#### POST /api/parse/pdf
Upload and parse a PDF file.

**Request:**
- `file` (multipart): PDF file
- `prompt_mode` (form): Parsing mode

**Response:** Same as image endpoint

#### GET /api/jobs/{job_id}/status
Get current job status and progress.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "processing",
  "file_type": "pdf",
  "total_pages": 10,
  "current_page": 3,
  "progress_percent": 30.0,
  "message": "Processing page 3/10...",
  "created_at": "2024-01-01T00:00:00",
  "updated_at": "2024-01-01T00:01:00"
}
```

#### GET /api/jobs/{job_id}/results
Get parsed results.

**Response:**
```json
{
  "job_id": "uuid",
  "status": "completed",
  "results": [...],
  "download_url": "/api/jobs/{job_id}/download"
}
```

#### GET /api/jobs/{job_id}/download
Download results as ZIP file.

**Response:** ZIP file download

#### WS /api/jobs/{job_id}/stream
WebSocket endpoint for real-time progress.

**Message Format:**
```json
{
  "event": "status_update",
  "data": { /* JobStatus object */ }
}
```

### Prompt Modes

- `prompt_layout_all_en`: Full layout detection + text (default)
- `prompt_layout_only_en`: Layout detection only (no text)
- `prompt_ocr`: Text extraction only (markdown)

---

## Support

For issues:
1. Check HF Spaces logs for backend errors
2. Check browser console for frontend errors
3. Verify environment variables are correct
4. Test API directly at `/docs` endpoint

---

## License

This project uses the dots.ocr model which has its own license. Check the model repository for details.
