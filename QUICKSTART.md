# Quick Start Guide

## What Changed?

Your DotsOCR application now has:
- ✅ **FastAPI backend** (replaces Gradio)
- ✅ **React frontend** (modern UI)
- ✅ **Separate deployment** (backend on HF Spaces, frontend on Hostinger/local)

## File Structure

```
dots-ocr-parser/
├── app.py                   # NEW: FastAPI backend
├── app_gradio_backup.py     # OLD: Gradio version (backup)
├── requirements.txt         # UPDATED: Added FastAPI dependencies
├── frontend/                # NEW: React application
│   ├── src/
│   │   ├── components/      # UI components
│   │   ├── api/             # Backend API client
│   │   └── types/           # TypeScript types
│   ├── .env.development     # Dev config
│   ├── .env.production      # Prod config
│   └── package.json
├── DEPLOYMENT.md            # Detailed deployment guide
├── README_API.md            # API documentation
└── QUICKSTART.md            # This file
```

## Step-by-Step Deployment

### Step 1: Deploy Backend to HuggingFace Spaces

```bash
# 1. Make sure you're in the project root
cd /Users/mac/Desktop/dots-ocr-parser

# 2. Add your HF Spaces as remote (if not already added)
git remote add hf https://huggingface.co/spaces/isseygino911/dots-ocr-parser

# 3. Commit the new FastAPI backend
git add app.py requirements.txt .gitignore
git commit -m "Switch from Gradio to FastAPI backend"

# 4. Push to HuggingFace Spaces
git push hf main

# 5. Wait for deployment (~10-15 minutes for first time)
```

**What happens:**
- HF Spaces will install dependencies
- Download the DotsOCR model (~3GB)
- Start the FastAPI server on port 7860
- Your API will be available at: `https://isseygino911-dots-ocr-parser.hf.space`

**Verify it worked:**
- Visit: `https://isseygino911-dots-ocr-parser.hf.space/docs`
- You should see the FastAPI automatic API documentation

### Step 2: Test Backend Locally (Optional)

You can't run the backend on your Mac (no CUDA), but you can test it via the API:

```bash
# Test health endpoint
curl https://isseygino911-dots-ocr-parser.hf.space/health

# Expected response:
# {"status":"healthy","gpu_available":true}
```

### Step 3: Run Frontend Locally

```bash
# 1. Navigate to frontend directory
cd /Users/mac/Desktop/dots-ocr-parser/frontend

# 2. Update backend URL in .env.development
echo "VITE_API_BASE_URL=https://isseygino911-dots-ocr-parser.hf.space" > .env.development

# 3. Install dependencies (if not already done)
npm install

# 4. Start dev server
npm run dev
```

**Access the app:**
- Open browser: `http://localhost:5173`
- Try uploading an image or PDF
- Watch real-time progress
- View results

### Step 4: Deploy Frontend to Hostinger

```bash
# 1. Build production version
cd /Users/mac/Desktop/dots-ocr-parser/frontend

# 2. Set production backend URL
echo "VITE_API_BASE_URL=https://isseygino911-dots-ocr-parser.hf.space" > .env.production

# 3. Build
npm run build

# 4. Upload the dist/ folder to Hostinger
# - Use FTP client (FileZilla, etc.)
# - Or use Hostinger File Manager
# - Upload contents of dist/ to your public_html folder
```

**Access your deployed app:**
- Visit: `https://yourdomain.com`

---

## Testing the Complete Flow

### 1. Upload an Image

1. Open the frontend (local or deployed)
2. Drag and drop an image or click to select
3. Choose parsing mode (default is fine)
4. Click upload or let it auto-upload

### 2. Watch Progress

- You'll see real-time progress updates
- Progress bar shows percentage
- Messages show current processing stage

### 3. View Results

Once complete:
- **Markdown Preview**: Rendered document content
- **Raw Text**: Plain markdown
- **JSON**: Structured data with bboxes

For PDFs:
- Use Previous/Next buttons to navigate pages
- Each page has its own results

### 4. Download Results

Click "Download ZIP" to get:
- All page images with layout visualization
- JSON files with structured data
- Markdown files with text content

---

## Common Issues & Solutions

### Issue: "Failed to fetch" error

**Cause:** Backend URL is wrong or backend is down

**Solution:**
1. Check backend is running: Visit `https://isseygino911-dots-ocr-parser.hf.space/health`
2. Verify `.env.development` or `.env.production` has correct URL
3. Check browser console for CORS errors

### Issue: CORS errors in browser console

**Cause:** Backend doesn't allow your frontend domain

**Solution:**
Update `app.py` line 199:
```python
allow_origins=["https://yourdomain.com", "http://localhost:5173"],
```
Then push to HF Spaces again.

### Issue: WebSocket connection failed

**Don't worry!** The app automatically falls back to HTTP polling. Everything will still work, just without real-time updates.

### Issue: Backend returns 503 or times out

**Cause:** HF Space is sleeping or overloaded

**Solution:**
1. Visit the Space URL to wake it up
2. Wait 30 seconds for model to load
3. Try again

### Issue: "Job not found" after backend restart

**Cause:** Jobs are stored in memory, lost on restart

**Solution:** Upload the file again. For persistent storage, see DEPLOYMENT.md for SQLite enhancement.

---

## API Endpoints Reference

### Upload Image
```bash
curl -X POST \
  https://isseygino911-dots-ocr-parser.hf.space/api/parse/image \
  -F "file=@test.jpg" \
  -F "prompt_mode=prompt_layout_all_en"
```

### Check Status
```bash
curl https://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/status
```

### Get Results
```bash
curl https://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/results
```

### Download ZIP
```bash
curl -O https://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/download
```

---

## Configuration Options

### Parsing Modes

- **prompt_layout_all_en** (Default)
  - Full layout detection + text recognition
  - Output: JSON with bboxes, categories, and text
  - Best for: Complete document analysis

- **prompt_layout_only_en**
  - Layout detection only (no text)
  - Output: JSON with bboxes and categories
  - Best for: Document structure analysis

- **prompt_ocr**
  - Text extraction only
  - Output: Plain markdown
  - Best for: Quick text extraction

### Frontend Environment Variables

**Development** (`.env.development`):
```bash
VITE_API_BASE_URL=http://localhost:7860  # or HF Spaces URL for testing
```

**Production** (`.env.production`):
```bash
VITE_API_BASE_URL=https://isseygino911-dots-ocr-parser.hf.space
```

---

## Performance Expectations

- **Backend first start**: 10-15 minutes (model download)
- **Backend subsequent starts**: 30-60 seconds (model cached)
- **Image processing**: 10-15 seconds per image
- **PDF processing**: 10-15 seconds per page
- **Frontend load time**: < 2 seconds

---

## Next Steps

### Option 1: Basic Usage
You're done! Start using the app.

### Option 2: Enhancements
See [DEPLOYMENT.md](./DEPLOYMENT.md) for:
- Adding authentication (JWT)
- Adding rate limiting
- Persistent job storage (SQLite)
- Monitoring and logging

### Option 3: Customization
Edit frontend components:
- `frontend/src/components/FileUpload.tsx` - Upload UI
- `frontend/src/components/JobProgress.tsx` - Progress display
- `frontend/src/components/ResultsViewer.tsx` - Results display
- `frontend/src/App.tsx` - Main layout

---

## Getting Help

1. **Check logs:**
   - Backend: HuggingFace Spaces logs
   - Frontend: Browser DevTools console

2. **Read docs:**
   - [DEPLOYMENT.md](./DEPLOYMENT.md) - Detailed deployment guide
   - [README_API.md](./README_API.md) - API documentation

3. **Test API directly:**
   - Visit `/docs` on your backend URL
   - Try endpoints in the interactive docs

---

## Rollback to Gradio (if needed)

If you want to go back to the original Gradio UI:

```bash
# Restore backup
cp app_gradio_backup.py app.py

# Push to HF Spaces
git add app.py
git commit -m "Rollback to Gradio"
git push hf main
```

---

## Summary Checklist

- [ ] Backend deployed to HuggingFace Spaces
- [ ] Backend URL is accessible at `/docs`
- [ ] Frontend `.env` file has correct backend URL
- [ ] Frontend runs locally (`npm run dev`)
- [ ] Can upload image/PDF successfully
- [ ] Can see real-time progress
- [ ] Can view results in all tabs
- [ ] Can download results as ZIP
- [ ] Frontend deployed to Hostinger (optional)

Once all checkboxes are done, you're all set!

---

**Enjoy your new React + FastAPI DotsOCR application!** 🎉
