#!/usr/bin/env python3
import subprocess
import sys
import os
import uuid
import json
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, List
import asyncio
from datetime import datetime

# Install dependencies
print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn==2.8.0.post2", "--no-build-isolation"], check=True)

if not os.path.exists("dots.ocr"):
    print("Cloning dots.ocr repository...")
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

print("Installing dots.ocr...")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr"], check=True)

print("Downloading model...")
os.makedirs("./dots.ocr/weights", exist_ok=True)
model_path = os.path.abspath("./dots.ocr/weights/DotsOCR")

if not os.path.exists(model_path):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="rednote-hilab/dots.ocr",
        local_dir=model_path,
        local_dir_use_symlinks=False
    )

print(f"Model downloaded to: {model_path}")

os.chdir("./dots.ocr")

# Patch DotsOCRParser with progress tracking
print("Patching parser with progress tracking...")
sys.path.insert(0, os.getcwd())

import dots_ocr.parser as parser_module

original_load = parser_module.DotsOCRParser._load_hf_model

def patched_load(self):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available!")

    model_path = os.path.abspath("./weights/DotsOCR")
    print(f"Loading model from: {model_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    self.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    self.model.eval()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    self.processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    self.process_vision_info = process_vision_info

    print(f"✅ Model loaded")

parser_module.DotsOCRParser._load_hf_model = patched_load

# Optimize inference
original_inference = parser_module.DotsOCRParser._inference_with_hf

def optimized_inference(self, image, prompt):
    import torch
    import time

    start = time.time()
    print(f"🔄 Starting inference...")

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = self.process_vision_info(messages)
    inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    print(f"⏱️  Preprocessing: {time.time()-start:.1f}s")
    gen_start = time.time()

    with torch.inference_mode():
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=6000,
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )

    print(f"⏱️  Generation: {time.time()-gen_start:.1f}s")

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    torch.cuda.empty_cache()

    print(f"✅ Total: {time.time()-start:.1f}s")

    return response

parser_module.DotsOCRParser._inference_with_hf = optimized_inference

# Store progress callbacks globally
progress_callbacks = {}

# Patch PDF processing to support callbacks
original_parse_pdf = parser_module.DotsOCRParser.parse_pdf

def parse_pdf_with_progress(self, input_path, filename, prompt_mode, save_dir):
    import time
    print(f"\n{'='*60}")
    print(f"📄 Starting PDF processing: {input_path}")
    print(f"{'='*60}\n")

    from dots_ocr.utils.doc_utils import load_images_from_pdf

    total_start = time.time()
    print(f"📖 Loading PDF pages...")
    images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
    total_pages = len(images_origin)
    print(f"✅ Loaded {total_pages} pages in {time.time()-total_start:.1f}s\n")

    results = []
    for i, image in enumerate(images_origin):
        page_start = time.time()
        print(f"\n{'─'*60}")
        print(f"📄 Processing page {i+1}/{total_pages}...")
        print(f"{'─'*60}")

        # Call progress callback if registered
        job_id = getattr(self, '_current_job_id', None)
        if job_id and job_id in progress_callbacks:
            callback = progress_callbacks[job_id]
            asyncio.run(callback(i + 1, total_pages, f"Processing page {i+1}/{total_pages}"))

        result = self._parse_single_image(
            origin_image=image,
            prompt_mode=prompt_mode,
            save_dir=save_dir,
            save_name=filename,
            source="pdf",
            page_idx=i,
        )
        result['file_path'] = input_path
        results.append(result)

        elapsed = time.time() - page_start
        remaining = (total_pages - i - 1) * elapsed
        print(f"✅ Page {i+1}/{total_pages} done in {elapsed:.1f}s")
        print(f"⏳ Estimated remaining: {remaining:.0f}s ({remaining/60:.1f} min)\n")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✅ PDF COMPLETE: {total_pages} pages in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"📊 Average: {total_time/total_pages:.1f}s per page")
    print(f"{'='*60}\n")

    return results

parser_module.DotsOCRParser.parse_pdf = parse_pdf_with_progress

# Initialize FastAPI
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="DotsOCR API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage paths
DATA_DIR = Path("/data") if os.path.exists("/data") else Path("./data")  # HF Spaces uses /data
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Job storage (in-memory, could be replaced with SQLite)
jobs: Dict[str, Dict] = {}

# WebSocket connections
active_connections: Dict[str, List[WebSocket]] = {}

# Initialize parser
from dots_ocr.parser import DotsOCRParser

DEFAULT_CONFIG = {
    'ip': '0.0.0.0',
    'port_vllm': 8000,
    'min_pixels': 3136,
    'max_pixels': 11289600,
}

dots_parser = DotsOCRParser(
    ip=DEFAULT_CONFIG['ip'],
    port=DEFAULT_CONFIG['port_vllm'],
    dpi=200,
    min_pixels=DEFAULT_CONFIG['min_pixels'],
    max_pixels=DEFAULT_CONFIG['max_pixels'],
    use_hf=True
)

# Pydantic models
class JobStatus(BaseModel):
    job_id: str
    status: str
    file_type: str
    total_pages: Optional[int] = None
    current_page: Optional[int] = None
    progress_percent: float
    message: str
    created_at: str
    updated_at: str
    error: Optional[str] = None

class JobResult(BaseModel):
    job_id: str
    status: str
    results: Optional[Dict] = None
    download_url: Optional[str] = None

# Helper functions
async def update_job_status(job_id: str, status: str, **kwargs):
    """Update job status and notify WebSocket clients"""
    if job_id in jobs:
        jobs[job_id]['status'] = status
        jobs[job_id]['updated_at'] = datetime.now().isoformat()
        jobs[job_id].update(kwargs)

        # Notify WebSocket clients
        if job_id in active_connections:
            message = {
                "event": "status_update",
                "data": jobs[job_id]
            }
            for connection in active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except:
                    pass

async def progress_callback(job_id: str):
    """Create a progress callback for a specific job"""
    async def callback(current_page: int, total_pages: int, message: str):
        progress_percent = (current_page / total_pages) * 100
        await update_job_status(
            job_id,
            status="processing",
            current_page=current_page,
            total_pages=total_pages,
            progress_percent=progress_percent,
            message=message
        )
    return callback

def process_document(job_id: str, file_path: Path, file_type: str, prompt_mode: str = "prompt_layout_all_en"):
    """Process document (image or PDF) synchronously"""
    try:
        job = jobs[job_id]
        result_dir = RESULTS_DIR / job_id
        result_dir.mkdir(parents=True, exist_ok=True)

        # Set job_id on parser for progress tracking
        dots_parser._current_job_id = job_id

        # Register progress callback
        progress_callbacks[job_id] = lambda c, t, m: asyncio.run(update_job_status(
            job_id, "processing", current_page=c, total_pages=t, progress_percent=(c/t)*100, message=m
        ))

        # Update status to processing
        asyncio.run(update_job_status(
            job_id,
            status="processing",
            progress_percent=0,
            message="Starting document processing..."
        ))

        # Process file - specify output_dir to save results in job directory
        results = dots_parser.parse_file(
            input_path=str(file_path),
            output_dir=str(result_dir),
            prompt_mode=prompt_mode,
            bbox=None
        )

        # Clean up callback
        if job_id in progress_callbacks:
            del progress_callbacks[job_id]

        # Format results for API response
        filename = file_path.stem
        parsed_output_dir = result_dir / filename

        # Build structured results
        formatted_results = {
            "pages": []
        }

        for idx, result in enumerate(results):
            page_num = idx + 1
            page_data = {
                "page_number": page_num,
                "markdown": result.get("markdown", ""),
                "json_output": result,
                "annotated_image_path": f"/api/results/{job_id}/{filename}/page_{page_num}.png"
            }
            formatted_results["pages"].append(page_data)

        # Store formatted results
        job['results'] = formatted_results
        job['result_dir'] = str(result_dir)
        job['parsed_output_dir'] = str(parsed_output_dir)

        # Create download package - zip the actual parsed output directory
        if parsed_output_dir.exists():
            zip_base_path = result_dir / f"{job_id}_results"
            shutil.make_archive(str(zip_base_path), 'zip', str(parsed_output_dir))
        else:
            print(f"Warning: Parsed output directory not found: {parsed_output_dir}")

        asyncio.run(update_job_status(
            job_id,
            status="completed",
            progress_percent=100,
            message="Processing completed successfully",
            download_url=f"/api/jobs/{job_id}/download"
        ))

    except Exception as e:
        print(f"Error processing job {job_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        asyncio.run(update_job_status(
            job_id,
            status="failed",
            message=f"Processing failed: {str(e)}",
            error=str(e)
        ))

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "DotsOCR API",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "gpu_available": True}

@app.post("/api/parse/image")
async def parse_image(
    file: UploadFile = File(...),
    prompt_mode: str = Form("prompt_layout_all_en")
):
    """Upload and parse a single image"""
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    # Create job
    job_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / job_id / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "file_type": "image",
        "filename": file.filename,
        "file_path": str(file_path),
        "prompt_mode": prompt_mode,
        "total_pages": 1,
        "current_page": 0,
        "progress_percent": 0,
        "message": "Job queued",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    # Process in background (in real app, use task queue)
    import threading
    thread = threading.Thread(target=process_document, args=(job_id, file_path, "image", prompt_mode))
    thread.start()

    return {"job_id": job_id, "status": "queued"}

@app.post("/api/parse/pdf")
async def parse_pdf(
    file: UploadFile = File(...),
    prompt_mode: str = Form("prompt_layout_all_en")
):
    """Upload and parse a PDF document"""
    # Validate file type
    if file.content_type != "application/pdf":
        raise HTTPException(400, "File must be a PDF")

    # Create job
    job_id = str(uuid.uuid4())
    file_path = UPLOADS_DIR / job_id / file.filename
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "file_type": "pdf",
        "filename": file.filename,
        "file_path": str(file_path),
        "prompt_mode": prompt_mode,
        "total_pages": None,  # Will be determined during processing
        "current_page": 0,
        "progress_percent": 0,
        "message": "Job queued",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    # Process in background
    import threading
    thread = threading.Thread(target=process_document, args=(job_id, file_path, "pdf", prompt_mode))
    thread.start()

    return {"job_id": job_id, "status": "queued"}

@app.get("/api/jobs/{job_id}/status")
async def get_job_status(job_id: str):
    """Get job status and progress"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    return jobs[job_id]

@app.get("/api/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Get job results"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]

    if job['status'] != "completed":
        return {
            "job_id": job_id,
            "status": job['status'],
            "message": job.get('message', ''),
            "results": None
        }

    return {
        "job_id": job_id,
        "status": "completed",
        "results": job.get('results'),
        "download_url": f"/api/jobs/{job_id}/download"
    }

@app.get("/api/jobs/{job_id}/download")
async def download_results(job_id: str):
    """Download results as ZIP file"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]

    if job['status'] != "completed":
        raise HTTPException(400, "Job not completed yet")

    result_dir = RESULTS_DIR / job_id
    zip_path = result_dir / f"{job_id}_results.zip"

    if not zip_path.exists():
        raise HTTPException(404, "Results file not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"results_{job['filename']}.zip"
    )

@app.websocket("/api/jobs/{job_id}/stream")
async def stream_progress(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time progress updates"""
    await websocket.accept()

    # Register connection
    if job_id not in active_connections:
        active_connections[job_id] = []
    active_connections[job_id].append(websocket)

    try:
        # Send current status immediately
        if job_id in jobs:
            await websocket.send_json({
                "event": "status_update",
                "data": jobs[job_id]
            })

        # Keep connection alive
        while True:
            try:
                # Wait for messages (client can send ping)
                data = await websocket.receive_text()

                # Send current status on ping
                if data == "ping" and job_id in jobs:
                    await websocket.send_json({
                        "event": "status_update",
                        "data": jobs[job_id]
                    })
            except WebSocketDisconnect:
                break

    finally:
        # Cleanup connection
        if job_id in active_connections:
            active_connections[job_id].remove(websocket)
            if not active_connections[job_id]:
                del active_connections[job_id]

@app.get("/api/results/{job_id}/{filename:path}")
async def get_result_file(job_id: str, filename: str):
    """Serve individual result files (images, JSON, markdown)"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    result_dir = RESULTS_DIR / job_id
    file_path = result_dir / filename

    # Security check: ensure the path is within result_dir
    try:
        file_path = file_path.resolve()
        result_dir = result_dir.resolve()
        if not str(file_path).startswith(str(result_dir)):
            raise HTTPException(403, "Access denied")
    except:
        raise HTTPException(404, "File not found")

    if not file_path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(file_path)

# Run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"\n{'='*60}")
    print(f"🚀 Starting DotsOCR API Server on port {port}")
    print(f"📚 API Documentation: http://0.0.0.0:{port}/docs")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
