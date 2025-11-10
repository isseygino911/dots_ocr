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
        # Check for cancellation before starting this page
        job_id = getattr(self, '_current_job_id', None)
        if job_id and job_id in cancelled_jobs:
            print(f"⚠️  Job {job_id} cancelled, stopping PDF processing")
            raise Exception("Job cancelled by user")

        page_start = time.time()
        print(f"\n{'─'*60}")
        print(f"📄 Processing page {i+1}/{total_pages}...")
        print(f"{'─'*60}")

        # Call progress callback if registered
        if job_id and job_id in progress_callbacks:
            callback = progress_callbacks[job_id]
            callback(i + 1, total_pages, f"Processing page {i+1}/{total_pages}")

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

# Cancellation flags
cancelled_jobs: set = set()

# Custom prompts storage (in-memory, could be replaced with database)
custom_prompts: Dict[str, Dict[str, str]] = {}  # {prompt_id: {"name": "...", "content": "..."}}

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
def update_job_status_sync(job_id: str, status: str, **kwargs):
    """Synchronous version - Update job status without WebSocket notifications"""
    if job_id in jobs:
        jobs[job_id]['status'] = status
        jobs[job_id]['updated_at'] = datetime.now().isoformat()
        jobs[job_id].update(kwargs)

async def update_job_status(job_id: str, status: str, **kwargs):
    """Async version - Update job status and notify WebSocket clients"""
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

        # Check if job was cancelled before starting
        if job_id in cancelled_jobs:
            update_job_status_sync(
                job_id,
                status="cancelled",
                message="Job was cancelled before processing started"
            )
            cancelled_jobs.discard(job_id)
            return

        # Set job_id on parser for progress tracking
        dots_parser._current_job_id = job_id

        # Register progress callback (use sync version in thread) with cancellation check
        def progress_with_cancel_check(c, t, m):
            if job_id in cancelled_jobs:
                raise Exception("Job cancelled by user")
            update_job_status_sync(
                job_id, "processing", current_page=c, total_pages=t, progress_percent=(c/t)*100, message=m
            )

        progress_callbacks[job_id] = progress_with_cancel_check

        # Update status to processing
        update_job_status_sync(
            job_id,
            status="processing",
            progress_percent=0,
            message="Starting document processing..."
        )

        # Determine actual prompt to use
        # If it's a custom prompt, we need to temporarily add it to the prompts dict
        from dots_ocr.utils import prompts as prompts_module

        actual_prompt_mode = prompt_mode
        temp_prompt_added = False

        if prompt_mode.startswith("custom_") and prompt_mode in custom_prompts:
            # Add custom prompt to the module's dict temporarily
            custom_content = custom_prompts[prompt_mode]["content"]
            prompts_module.dict_promptmode_to_prompt[prompt_mode] = custom_content
            temp_prompt_added = True

        try:
            # Process file - specify output_dir to save results in job directory
            results = dots_parser.parse_file(
                input_path=str(file_path),
                output_dir=str(result_dir),
                prompt_mode=actual_prompt_mode,
                bbox=None
            )
        finally:
            # Clean up temp prompt if added
            if temp_prompt_added and prompt_mode in prompts_module.dict_promptmode_to_prompt:
                del prompts_module.dict_promptmode_to_prompt[prompt_mode]

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

            # Read markdown content from file
            markdown_content = ""
            if 'md_content_path' in result and os.path.exists(result['md_content_path']):
                with open(result['md_content_path'], 'r', encoding='utf-8') as f:
                    markdown_content = f.read()

            # Read JSON layout info from file
            json_data = result  # Keep full result as fallback
            if 'layout_info_path' in result and os.path.exists(result['layout_info_path']):
                with open(result['layout_info_path'], 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

            # Get actual image filename (DotsOCR saves as .jpg, not .png)
            # For PDFs: page_1.jpg, page_2.jpg, etc.
            # For images: filename.jpg
            if file_type == "pdf":
                image_filename = f"page_{page_num}.jpg"
            else:
                image_filename = f"{filename}.jpg"

            page_data = {
                "page_number": page_num,
                "markdown": markdown_content,
                "json_output": json_data,
                "annotated_image_path": f"/api/results/{job_id}/{filename}/{image_filename}"
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

        update_job_status_sync(
            job_id,
            status="completed",
            progress_percent=100,
            message="Processing completed successfully",
            download_url=f"/api/jobs/{job_id}/download"
        )

    except Exception as e:
        # Check if this was a cancellation
        if "cancelled by user" in str(e).lower() or job_id in cancelled_jobs:
            print(f"Job {job_id} cancelled by user")
            update_job_status_sync(
                job_id,
                status="cancelled",
                message="Job cancelled by user"
            )
            cancelled_jobs.discard(job_id)
        else:
            print(f"Error processing job {job_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            update_job_status_sync(
                job_id,
                status="failed",
                message=f"Processing failed: {str(e)}",
                error=str(e)
            )
    finally:
        # Clean up
        if job_id in progress_callbacks:
            del progress_callbacks[job_id]
        cancelled_jobs.discard(job_id)

# API Endpoints

@app.get("/")
async def root():
    """
    API Root - Welcome endpoint

    Returns basic API information and links to documentation.
    """
    return {
        "message": "DotsOCR API",
        "docs": "/docs",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """
    Health Check

    Check if the API is running and GPU is available.

    Returns:
        - status: "healthy" if the API is running
        - gpu_available: true if CUDA GPU is available

    Example:
        ```bash
        curl https://isseygino911-dots-ocr-parser.hf.space/health
        ```
    """
    return {"status": "healthy", "gpu_available": True}

@app.post("/api/parse/image")
async def parse_image(
    file: UploadFile = File(..., description="Image file to parse"),
    prompt_mode: str = Form("prompt_layout_all_en", description="Parsing mode: prompt_layout_all_en (full layout + text), prompt_layout_only_en (layout only), or prompt_ocr (text only)")
):
    """
    Parse Image - Extract text and layout from an image

    Upload an image file and parse it to extract text and layout information.
    The processing happens asynchronously - you'll receive a job_id to track progress.

    Supported formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .gif, .webp

    Parsing Modes:
        - **prompt_layout_all_en** (default): Full layout detection + text extraction
        - **prompt_layout_only_en**: Layout detection only, no text extraction
        - **prompt_ocr**: Text extraction only, no layout detection

    Returns:
        - job_id: Unique identifier to track this job
        - status: "queued" (processing will start immediately)

    Example (cURL):
        ```bash
        curl -X POST https://isseygino911-dots-ocr-parser.hf.space/api/parse/image \\
          -F "file=@document.jpg" \\
          -F "prompt_mode=prompt_layout_all_en"
        ```

    Example (Python):
        ```python
        import requests

        with open("document.jpg", "rb") as f:
            response = requests.post(
                "https://isseygino911-dots-ocr-parser.hf.space/api/parse/image",
                files={"file": f},
                data={"prompt_mode": "prompt_layout_all_en"}
            )

        job_id = response.json()["job_id"]
        print(f"Job ID: {job_id}")
        ```

    Example (JavaScript):
        ```javascript
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('prompt_mode', 'prompt_layout_all_en');

        const response = await fetch(
            'https://isseygino911-dots-ocr-parser.hf.space/api/parse/image',
            { method: 'POST', body: formData }
        );

        const { job_id } = await response.json();
        console.log('Job ID:', job_id);
        ```

    Next Steps:
        1. Use GET /api/jobs/{job_id}/status to monitor progress
        2. Use GET /api/jobs/{job_id}/results to get parsed data
        3. Use GET /api/jobs/{job_id}/download to download ZIP
    """
    # Validate file type by extension (more reliable than content_type)
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp'}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"File must be an image. Supported formats: {', '.join(allowed_extensions)}")

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
    file: UploadFile = File(..., description="PDF file to parse"),
    prompt_mode: str = Form("prompt_layout_all_en", description="Parsing mode for all pages")
):
    """
    Parse PDF - Extract text and layout from a PDF document

    Upload a PDF file and parse all pages to extract text and layout information.
    Each page is processed sequentially with real-time progress updates.

    Parsing Modes:
        - **prompt_layout_all_en** (default): Full layout detection + text extraction
        - **prompt_layout_only_en**: Layout detection only, no text extraction
        - **prompt_ocr**: Text extraction only, no layout detection

    Processing Time:
        - ~10-15 seconds per page
        - Progress updates available via WebSocket or polling

    Returns:
        - job_id: Unique identifier to track this job
        - status: "queued" (processing will start immediately)

    Example (cURL):
        ```bash
        curl -X POST https://isseygino911-dots-ocr-parser.hf.space/api/parse/pdf \\
          -F "file=@document.pdf" \\
          -F "prompt_mode=prompt_layout_all_en"
        ```

    Example (Python):
        ```python
        import requests
        import time

        with open("document.pdf", "rb") as f:
            response = requests.post(
                "https://isseygino911-dots-ocr-parser.hf.space/api/parse/pdf",
                files={"file": f},
                data={"prompt_mode": "prompt_layout_all_en"}
            )

        job_id = response.json()["job_id"]

        # Poll for completion
        while True:
            status = requests.get(
                f"https://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/status"
            ).json()

            print(f"Progress: {status['progress_percent']:.0f}%")

            if status['status'] == 'completed':
                break

            time.sleep(2)
        ```

    Next Steps:
        1. Use GET /api/jobs/{job_id}/status to monitor progress
        2. Use WS /api/jobs/{job_id}/stream for real-time updates
        3. Use GET /api/jobs/{job_id}/results to get all pages
        4. Use GET /api/jobs/{job_id}/download to download ZIP
    """
    # Validate file type by extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext != '.pdf':
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
    """
    Get Job Status - Monitor processing progress

    Check the current status and progress of a parsing job.
    Poll this endpoint every 2-3 seconds to monitor progress.

    Status Values:
        - **queued**: Job is waiting to be processed
        - **processing**: Currently processing (see progress_percent for progress)
        - **completed**: Processing finished successfully
        - **failed**: An error occurred (see error field)

    Returns:
        - job_id: The job identifier
        - status: Current job status
        - progress_percent: Progress from 0 to 100
        - message: Current processing message
        - current_page: Current page being processed (for PDFs)
        - total_pages: Total number of pages (for PDFs)
        - created_at: Job creation timestamp
        - updated_at: Last update timestamp
        - error: Error message (only if status is "failed")

    Example (Python):
        ```python
        import requests
        import time

        job_id = "your-job-id"

        while True:
            response = requests.get(
                f"https://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/status"
            )
            status = response.json()

            print(f"Status: {status['status']} - {status['progress_percent']:.0f}%")

            if status['status'] in ['completed', 'failed']:
                break

            time.sleep(2)
        ```

    Example (JavaScript):
        ```javascript
        async function pollStatus(jobId) {
            while (true) {
                const response = await fetch(
                    `https://isseygino911-dots-ocr-parser.hf.space/api/jobs/${jobId}/status`
                );
                const status = await response.json();

                console.log(`Progress: ${status.progress_percent}%`);

                if (status.status === 'completed' || status.status === 'failed') {
                    break;
                }

                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }
        ```

    Alternative: Use WebSocket at /api/jobs/{job_id}/stream for real-time updates
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    return jobs[job_id]

@app.get("/api/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """
    Get Job Results - Retrieve parsed document data

    Get the complete parsing results including extracted text, layout information,
    and bounding boxes for all detected elements.

    Returns (when completed):
        - job_id: The job identifier
        - status: "completed"
        - results: Parsed data with structure:
            - pages: Array of page results
                - page_number: Page index (1-based)
                - markdown: Extracted text in markdown format
                - json_output: Structured data with:
                    - bboxes: Bounding box coordinates [[x1,y1,x2,y2], ...]
                    - labels: Element types ["title", "paragraph", "table", ...]
                    - pred_text: Extracted text for each element
                - annotated_image_path: URL to view annotated image
        - download_url: URL to download all results as ZIP

    Returns (when not completed):
        - job_id: The job identifier
        - status: Current status ("queued", "processing", or "failed")
        - message: Status message
        - results: null

    Example (Python):
        ```python
        import requests

        response = requests.get(
            f"https://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/results"
        )
        data = response.json()

        if data['status'] == 'completed':
            for page in data['results']['pages']:
                print(f"Page {page['page_number']}:")
                print(page['markdown'][:200])  # First 200 chars
                print(f"Detected {len(page['json_output']['labels'])} elements")
        ```

    Example (JavaScript):
        ```javascript
        const response = await fetch(
            `https://isseygino911-dots-ocr-parser.hf.space/api/jobs/${jobId}/results`
        );
        const data = await response.json();

        if (data.status === 'completed') {
            data.results.pages.forEach(page => {
                console.log(`Page ${page.page_number}:`);
                console.log(page.markdown.substring(0, 200));
            });
        }
        ```
    """
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
    """
    Download Results - Get all results as a ZIP file

    Download a ZIP archive containing all parsing results including:
    - Annotated images with bounding boxes drawn (PNG files)
    - JSON files with structured data (bboxes, labels, text)
    - Markdown files with extracted text
    - JSONL file with all results

    The ZIP file is only available after the job status is "completed".

    Returns:
        ZIP file download with name: results_{filename}.zip

    Example (Python - download to file):
        ```python
        import requests

        response = requests.get(
            f"https://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/download"
        )

        with open("results.zip", "wb") as f:
            f.write(response.content)

        print("Downloaded results.zip")
        ```

    Example (JavaScript - trigger browser download):
        ```javascript
        const downloadUrl =
            `https://isseygino911-dots-ocr-parser.hf.space/api/jobs/${jobId}/download`;

        // Option 1: Open in new tab (triggers download)
        window.open(downloadUrl, '_blank');

        // Option 2: Use download link
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = 'results.zip';
        link.click();
        ```

    Example (cURL):
        ```bash
        curl -O https://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/download
        ```

    ZIP Contents:
        - page_1.png, page_2.png, ... (annotated images)
        - page_1.json, page_2.json, ... (structured data)
        - page_1.md, page_2.md, ... (markdown text)
        - {filename}.jsonl (all results in JSONL format)
    """
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

@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """
    Cancel Job - Stop a running processing job

    Cancel a job that is currently queued or processing. Once cancelled,
    the job status will change to "cancelled" and processing will stop.

    Note: Cancellation may take a few seconds to take effect, especially
    if a page is currently being processed. The job will be cancelled
    after the current page completes.

    Returns:
        - job_id: The job identifier
        - status: Current status after cancellation request
        - message: Confirmation message

    Example (Python):
        ```python
        import requests

        response = requests.post(
            f"https://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/cancel"
        )
        print(response.json())
        ```

    Example (JavaScript):
        ```javascript
        const response = await fetch(
            `https://isseygino911-dots-ocr-parser.hf.space/api/jobs/${jobId}/cancel`,
            { method: 'POST' }
        );
        const result = await response.json();
        console.log(result.message);
        ```

    Example (cURL):
        ```bash
        curl -X POST https://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/cancel
        ```
    """
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    current_status = job['status']

    # Can only cancel queued or processing jobs
    if current_status not in ['queued', 'processing']:
        return {
            "job_id": job_id,
            "status": current_status,
            "message": f"Cannot cancel job with status: {current_status}"
        }

    # Mark job for cancellation
    cancelled_jobs.add(job_id)

    # If job is still queued, mark it as cancelled immediately
    if current_status == 'queued':
        update_job_status_sync(
            job_id,
            status="cancelled",
            message="Job cancelled before processing started"
        )

    return {
        "job_id": job_id,
        "status": "cancelling",
        "message": "Cancellation requested. Job will be cancelled shortly."
    }

@app.websocket("/api/jobs/{job_id}/stream")
async def stream_progress(websocket: WebSocket, job_id: str):
    """
    WebSocket Stream - Real-time progress updates

    Connect via WebSocket to receive real-time progress updates for a job.
    More efficient than HTTP polling for long-running jobs (PDFs with many pages).

    Message Format (received from server):
        ```json
        {
            "event": "status_update",
            "data": {
                "job_id": "...",
                "status": "processing",
                "progress_percent": 50.0,
                "current_page": 5,
                "total_pages": 10,
                "message": "Processing page 5/10..."
            }
        }
        ```

    Client Actions:
        - Send "ping" message to request current status
        - Server automatically sends updates when progress changes
        - Connection closes when job completes or fails

    Example (JavaScript):
        ```javascript
        const ws = new WebSocket(
            'wss://isseygino911-dots-ocr-parser.hf.space/api/jobs/YOUR-JOB-ID/stream'
        );

        ws.onopen = () => {
            console.log('Connected');
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            const status = message.data;

            console.log(`Progress: ${status.progress_percent}%`);
            console.log(`Message: ${status.message}`);

            if (status.status === 'completed') {
                console.log('Processing complete!');
                ws.close();
            }
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            // Fallback to HTTP polling
        };

        // Optional: Send ping to request current status
        ws.send('ping');
        ```

    Example (Python with websockets):
        ```python
        import asyncio
        import websockets
        import json

        async def monitor_job(job_id):
            uri = f"wss://isseygino911-dots-ocr-parser.hf.space/api/jobs/{job_id}/stream"

            async with websockets.connect(uri) as websocket:
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    status = data['data']

                    print(f"Progress: {status['progress_percent']:.0f}%")

                    if status['status'] in ['completed', 'failed']:
                        break

        asyncio.run(monitor_job("your-job-id"))
        ```

    Fallback: If WebSocket connection fails, use GET /api/jobs/{job_id}/status with polling
    """
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

# ============================================================================
# Custom Prompt Management Endpoints
# ============================================================================

@app.get("/api/prompts")
async def list_prompts():
    """
    List All Prompts - Get default and custom prompts

    Returns all available prompts including both default (from DotsOCR library)
    and user-created custom prompts.

    Returns:
        - default_prompts: Array of default prompt objects
        - custom_prompts: Array of custom prompt objects

    Each prompt object contains:
        - id: Prompt identifier
        - name: Display name
        - content: Prompt text
        - is_custom: Boolean indicating if it's a custom prompt
    """
    from dots_ocr.utils.prompts import dict_promptmode_to_prompt

    # Default prompts from DotsOCR library
    default_prompt_names = {
        "prompt_layout_all_en": "Full Layout + Text (Default)",
        "prompt_layout_only_en": "Layout Only (No Text)",
        "prompt_ocr": "Text Only (Markdown)",
    }

    default_prompts = [
        {
            "id": prompt_id,
            "name": name,
            "content": dict_promptmode_to_prompt[prompt_id],
            "is_custom": False,
            "is_default": True
        }
        for prompt_id, name in default_prompt_names.items()
    ]

    # Custom prompts
    custom_prompt_list = [
        {
            "id": prompt_id,
            "name": data["name"],
            "content": data["content"],
            "is_custom": True,
            "is_default": False
        }
        for prompt_id, data in custom_prompts.items()
    ]

    return {
        "default_prompts": default_prompts,
        "custom_prompts": custom_prompt_list
    }

@app.post("/api/prompts")
async def create_custom_prompt(
    name: str = Form(..., description="Display name for the custom prompt"),
    content: str = Form(..., description="The prompt text content")
):
    """
    Create Custom Prompt - Save a new custom prompt

    Create and save a custom prompt that can be used for document parsing.
    The prompt will be available in the prompt selection dropdown.

    Args:
        - name: A display name for the prompt (e.g., "My Table Extractor")
        - content: The full prompt text

    Returns:
        - prompt_id: Unique identifier for the created prompt
        - name: Display name
        - content: Prompt text

    Example (cURL):
        ```bash
        curl -X POST https://isseygino911-dots-ocr-parser.hf.space/api/prompts \\
          -F "name=My Custom Prompt" \\
          -F "content=Extract all tables from this document..."
        ```

    Example (Python):
        ```python
        import requests

        response = requests.post(
            "https://isseygino911-dots-ocr-parser.hf.space/api/prompts",
            data={
                "name": "My Table Extractor",
                "content": "Extract all tables as HTML..."
            }
        )
        prompt_id = response.json()["prompt_id"]
        ```
    """
    # Generate unique ID
    prompt_id = f"custom_{str(uuid.uuid4())[:8]}"

    # Save custom prompt
    custom_prompts[prompt_id] = {
        "name": name,
        "content": content
    }

    return {
        "prompt_id": prompt_id,
        "name": name,
        "content": content,
        "is_custom": True
    }

@app.get("/api/prompts/{prompt_id}")
async def get_prompt(prompt_id: str):
    """
    Get Prompt Details - Retrieve a specific prompt

    Get the details of a specific prompt by its ID, including both
    default and custom prompts.

    Args:
        - prompt_id: The prompt identifier

    Returns:
        - id: Prompt identifier
        - name: Display name
        - content: Prompt text
        - is_custom: Boolean indicating if it's custom
    """
    from dots_ocr.utils.prompts import dict_promptmode_to_prompt

    # Check if it's a default prompt
    if prompt_id in dict_promptmode_to_prompt:
        default_names = {
            "prompt_layout_all_en": "Full Layout + Text (Default)",
            "prompt_layout_only_en": "Layout Only (No Text)",
            "prompt_ocr": "Text Only (Markdown)",
        }
        return {
            "id": prompt_id,
            "name": default_names.get(prompt_id, prompt_id),
            "content": dict_promptmode_to_prompt[prompt_id],
            "is_custom": False,
            "is_default": True
        }

    # Check if it's a custom prompt
    if prompt_id in custom_prompts:
        return {
            "id": prompt_id,
            "name": custom_prompts[prompt_id]["name"],
            "content": custom_prompts[prompt_id]["content"],
            "is_custom": True,
            "is_default": False
        }

    raise HTTPException(404, "Prompt not found")

@app.delete("/api/prompts/{prompt_id}")
async def delete_custom_prompt(prompt_id: str):
    """
    Delete Custom Prompt - Remove a custom prompt

    Delete a user-created custom prompt. Default prompts cannot be deleted.

    Args:
        - prompt_id: The custom prompt identifier to delete

    Returns:
        - message: Success message

    Example (cURL):
        ```bash
        curl -X DELETE https://isseygino911-dots-ocr-parser.hf.space/api/prompts/custom_abc123
        ```
    """
    # Prevent deletion of default prompts
    from dots_ocr.utils.prompts import dict_promptmode_to_prompt
    if prompt_id in dict_promptmode_to_prompt:
        raise HTTPException(403, "Cannot delete default prompts")

    # Delete custom prompt
    if prompt_id not in custom_prompts:
        raise HTTPException(404, "Custom prompt not found")

    del custom_prompts[prompt_id]

    return {"message": "Custom prompt deleted successfully", "prompt_id": prompt_id}

# Run server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    print(f"\n{'='*60}")
    print(f"🚀 Starting DotsOCR API Server on port {port}")
    print(f"📚 API Documentation: http://0.0.0.0:{port}/docs")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
