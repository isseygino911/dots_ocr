# DotsOCR API - Complete Usage Guide

## Base URL
```
https://isseygino911-dots-ocr-parser.hf.space
```

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [API Endpoints](#api-endpoints)
3. [Backend Examples (Python)](#backend-examples-python)
4. [Frontend Examples (JavaScript)](#frontend-examples-javascript)
5. [React Integration](#react-integration)
6. [Complete Workflows](#complete-workflows)
7. [Error Handling](#error-handling)

---

## Quick Start

### Test Your API (Command Line)
```bash
# 1. Health check
curl https://isseygino911-dots-ocr-parser.hf.space/health

# 2. Upload an image
curl -X POST https://isseygino911-dots-ocr-parser.hf.space/api/parse/image \
  -F "file=@image.jpg" \
  -F "prompt_mode=prompt_layout_all_en"

# Response: {"job_id": "abc-123-def", "status": "queued"}

# 3. Check status
curl https://isseygino911-dots-ocr-parser.hf.space/api/jobs/abc-123-def/status

# 4. Get results
curl https://isseygino911-dots-ocr-parser.hf.space/api/jobs/abc-123-def/results
```

---

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and GPU is available.

**Response:**
```json
{
  "status": "healthy",
  "gpu_available": true
}
```

---

### 2. Upload Image
**POST** `/api/parse/image`

Upload and parse an image file.

**Parameters:**
- `file` (required): Image file (multipart/form-data)
- `prompt_mode` (optional): Parsing mode
  - `prompt_layout_all_en` (default) - Full layout + text
  - `prompt_layout_only_en` - Layout only, no text
  - `prompt_ocr` - Text extraction only

**Supported Formats:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.gif`, `.webp`

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

---

### 3. Upload PDF
**POST** `/api/parse/pdf`

Upload and parse a PDF document (processes all pages).

**Parameters:**
- `file` (required): PDF file (multipart/form-data)
- `prompt_mode` (optional): Same as image endpoint

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued"
}
```

---

### 4. Get Job Status
**GET** `/api/jobs/{job_id}/status`

Monitor processing progress.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "file_type": "pdf",
  "filename": "document.pdf",
  "total_pages": 10,
  "current_page": 5,
  "progress_percent": 50.0,
  "message": "Processing page 5/10...",
  "created_at": "2024-01-01T10:00:00",
  "updated_at": "2024-01-01T10:02:30"
}
```

**Status values:**
- `queued` - Waiting to be processed
- `processing` - Currently processing
- `completed` - Successfully completed
- `failed` - Error occurred

---

### 5. Get Results
**GET** `/api/jobs/{job_id}/results`

Retrieve parsed document data.

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "results": {
    "pages": [
      {
        "page_number": 1,
        "markdown": "# Document Title\n\nThis is the extracted text...",
        "json_output": {
          "bboxes": [[100, 200, 500, 250], [100, 300, 500, 400]],
          "labels": ["title", "paragraph"],
          "pred_text": ["Document Title", "This is the extracted text..."]
        },
        "annotated_image_path": "/api/results/550e8400-.../test/page_1.png"
      }
    ]
  },
  "download_url": "/api/jobs/550e8400-e29b-41d4-a716-446655440000/download"
}
```

---

### 6. Download ZIP
**GET** `/api/jobs/{job_id}/download`

Download all results as a ZIP file.

**Response:** Binary ZIP file containing:
- Annotated images with bounding boxes (PNG)
- JSON files with structured data
- Markdown files with extracted text

---

### 7. WebSocket Progress Stream
**WS** `/api/jobs/{job_id}/stream`

Real-time progress updates via WebSocket.

**Message Format:**
```json
{
  "event": "status_update",
  "data": {
    "job_id": "550e8400-...",
    "status": "processing",
    "progress_percent": 50.0,
    "message": "Processing page 5/10..."
  }
}
```

---

## Backend Examples (Python)

### Basic Upload and Poll

```python
import requests
import time

API_BASE = "https://isseygino911-dots-ocr-parser.hf.space"

# 1. Upload file
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{API_BASE}/api/parse/pdf",
        files={"file": f},
        data={"prompt_mode": "prompt_layout_all_en"}
    )

job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")

# 2. Poll for completion
while True:
    status_resp = requests.get(f"{API_BASE}/api/jobs/{job_id}/status")
    status = status_resp.json()

    print(f"Progress: {status['progress_percent']:.0f}% - {status['message']}")

    if status["status"] == "completed":
        break
    elif status["status"] == "failed":
        print(f"Error: {status.get('error')}")
        break

    time.sleep(2)

# 3. Get results
results = requests.get(f"{API_BASE}/api/jobs/{job_id}/results").json()
print(f"Processed {len(results['results']['pages'])} pages")

# 4. Download ZIP
zip_response = requests.get(f"{API_BASE}/api/jobs/{job_id}/download")
with open("results.zip", "wb") as f:
    f.write(zip_response.content)
```

---

### Using WebSocket for Real-Time Updates

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
            print(f"Status: {status['status']} - {status['progress_percent']:.0f}%")

            if status['status'] in ['completed', 'failed']:
                break

# Usage
asyncio.run(monitor_job("your-job-id"))
```

---

### Flask Backend Integration

```python
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
API_BASE = "https://isseygino911-dots-ocr-parser.hf.space"

@app.route('/api/ocr', methods=['POST'])
def ocr_upload():
    # Receive file from your frontend
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Forward to DotsOCR API
    files = {'file': (file.filename, file.stream, file.content_type)}
    data = {'prompt_mode': request.form.get('prompt_mode', 'prompt_layout_all_en')}

    response = requests.post(
        f"{API_BASE}/api/parse/image",
        files=files,
        data=data
    )

    return jsonify(response.json())

@app.route('/api/ocr/status/<job_id>')
def ocr_status(job_id):
    response = requests.get(f"{API_BASE}/api/jobs/{job_id}/status")
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Frontend Examples (JavaScript)

### Vanilla JavaScript - Upload and Monitor

```html
<!DOCTYPE html>
<html>
<head>
    <title>DotsOCR Upload</title>
</head>
<body>
    <h1>Document OCR</h1>

    <input type="file" id="fileInput" accept="image/*,.pdf">
    <button onclick="uploadFile()">Upload</button>

    <div id="status"></div>
    <div id="results"></div>

    <script>
        const API_BASE = 'https://isseygino911-dots-ocr-parser.hf.space';

        async function uploadFile() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];

            if (!file) {
                alert('Please select a file');
                return;
            }

            // Determine endpoint based on file type
            const isPDF = file.name.toLowerCase().endsWith('.pdf');
            const endpoint = isPDF ? '/api/parse/pdf' : '/api/parse/image';

            // Upload file
            const formData = new FormData();
            formData.append('file', file);
            formData.append('prompt_mode', 'prompt_layout_all_en');

            const uploadResp = await fetch(`${API_BASE}${endpoint}`, {
                method: 'POST',
                body: formData
            });

            const { job_id } = await uploadResp.json();
            console.log('Job ID:', job_id);

            // Monitor progress
            monitorProgress(job_id);
        }

        async function monitorProgress(jobId) {
            const statusDiv = document.getElementById('status');

            while (true) {
                const response = await fetch(`${API_BASE}/api/jobs/${jobId}/status`);
                const status = await response.json();

                statusDiv.innerHTML = `
                    <p>Status: ${status.status}</p>
                    <p>Progress: ${status.progress_percent.toFixed(0)}%</p>
                    <p>${status.message}</p>
                `;

                if (status.status === 'completed') {
                    await displayResults(jobId);
                    break;
                } else if (status.status === 'failed') {
                    statusDiv.innerHTML += `<p style="color: red;">Error: ${status.error}</p>`;
                    break;
                }

                await new Promise(resolve => setTimeout(resolve, 2000));
            }
        }

        async function displayResults(jobId) {
            const response = await fetch(`${API_BASE}/api/jobs/${jobId}/results`);
            const data = await response.json();

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<h2>Results</h2>';

            data.results.pages.forEach(page => {
                resultsDiv.innerHTML += `
                    <div style="border: 1px solid #ccc; margin: 10px; padding: 10px;">
                        <h3>Page ${page.page_number}</h3>
                        <pre>${page.markdown.substring(0, 500)}...</pre>
                        <a href="${API_BASE}/api/jobs/${jobId}/download" download>
                            Download ZIP
                        </a>
                    </div>
                `;
            });
        }
    </script>
</body>
</html>
```

---

### Using Fetch API with Async/Await

```javascript
const API_BASE = 'https://isseygino911-dots-ocr-parser.hf.space';

// Upload file
async function uploadDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('prompt_mode', 'prompt_layout_all_en');

    const endpoint = file.name.endsWith('.pdf')
        ? '/api/parse/pdf'
        : '/api/parse/image';

    const response = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        body: formData
    });

    return await response.json();
}

// Check status
async function checkStatus(jobId) {
    const response = await fetch(`${API_BASE}/api/jobs/${jobId}/status`);
    return await response.json();
}

// Get results
async function getResults(jobId) {
    const response = await fetch(`${API_BASE}/api/jobs/${jobId}/results`);
    return await response.json();
}

// Complete workflow
async function processDocument(file) {
    try {
        // 1. Upload
        const { job_id } = await uploadDocument(file);
        console.log('Job ID:', job_id);

        // 2. Poll for completion
        while (true) {
            const status = await checkStatus(job_id);
            console.log(`Progress: ${status.progress_percent}%`);

            if (status.status === 'completed') {
                break;
            } else if (status.status === 'failed') {
                throw new Error(status.error);
            }

            await new Promise(resolve => setTimeout(resolve, 2000));
        }

        // 3. Get results
        const results = await getResults(job_id);
        console.log('Results:', results);

        return results;
    } catch (error) {
        console.error('Error:', error);
        throw error;
    }
}

// Usage
const fileInput = document.getElementById('fileInput');
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        const results = await processDocument(file);
        console.log('Processing complete!', results);
    }
});
```

---

### Using WebSocket for Real-Time Updates

```javascript
function monitorJobWithWebSocket(jobId, onUpdate, onComplete) {
    const ws = new WebSocket(
        `wss://isseygino911-dots-ocr-parser.hf.space/api/jobs/${jobId}/stream`
    );

    ws.onopen = () => {
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        const status = message.data;

        // Call update callback
        onUpdate(status);

        // Check if completed
        if (status.status === 'completed') {
            onComplete(status);
            ws.close();
        } else if (status.status === 'failed') {
            onComplete(status);
            ws.close();
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        // Fallback to HTTP polling
        pollJobStatus(jobId, onUpdate, onComplete);
    };

    return ws;
}

// Usage
const ws = monitorJobWithWebSocket(
    'job-id-here',
    (status) => {
        console.log(`Progress: ${status.progress_percent}%`);
        document.getElementById('progress').textContent =
            `${status.progress_percent.toFixed(0)}%`;
    },
    (status) => {
        console.log('Job completed!', status);
        displayResults(status.job_id);
    }
);
```

---

## React Integration

### Complete React Component

```jsx
import React, { useState, useCallback } from 'react';
import axios from 'axios';

const API_BASE = 'https://isseygino911-dots-ocr-parser.hf.space';

function DocumentOCR() {
    const [file, setFile] = useState(null);
    const [jobId, setJobId] = useState(null);
    const [status, setStatus] = useState(null);
    const [results, setResults] = useState(null);
    const [uploading, setUploading] = useState(false);

    // Handle file selection
    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
        setJobId(null);
        setStatus(null);
        setResults(null);
    };

    // Upload file
    const handleUpload = async () => {
        if (!file) return;

        setUploading(true);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('prompt_mode', 'prompt_layout_all_en');

        const endpoint = file.name.toLowerCase().endsWith('.pdf')
            ? '/api/parse/pdf'
            : '/api/parse/image';

        try {
            const response = await axios.post(`${API_BASE}${endpoint}`, formData);
            const { job_id } = response.data;
            setJobId(job_id);
            startPolling(job_id);
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Upload failed');
        } finally {
            setUploading(false);
        }
    };

    // Poll for status
    const startPolling = useCallback((jobId) => {
        const interval = setInterval(async () => {
            try {
                const response = await axios.get(`${API_BASE}/api/jobs/${jobId}/status`);
                const statusData = response.data;
                setStatus(statusData);

                if (statusData.status === 'completed') {
                    clearInterval(interval);
                    fetchResults(jobId);
                } else if (statusData.status === 'failed') {
                    clearInterval(interval);
                    alert('Processing failed: ' + statusData.error);
                }
            } catch (error) {
                console.error('Status check failed:', error);
            }
        }, 2000);

        return () => clearInterval(interval);
    }, []);

    // Fetch results
    const fetchResults = async (jobId) => {
        try {
            const response = await axios.get(`${API_BASE}/api/jobs/${jobId}/results`);
            setResults(response.data);
        } catch (error) {
            console.error('Failed to fetch results:', error);
        }
    };

    return (
        <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
            <h1>Document OCR</h1>

            {/* File Upload */}
            <div style={{ marginBottom: '20px' }}>
                <input
                    type="file"
                    accept="image/*,.pdf"
                    onChange={handleFileChange}
                    disabled={uploading}
                />
                <button
                    onClick={handleUpload}
                    disabled={!file || uploading}
                    style={{ marginLeft: '10px' }}
                >
                    {uploading ? 'Uploading...' : 'Upload'}
                </button>
            </div>

            {/* Status */}
            {status && (
                <div style={{
                    padding: '15px',
                    background: '#f0f0f0',
                    borderRadius: '5px',
                    marginBottom: '20px'
                }}>
                    <h3>Status: {status.status}</h3>
                    <div style={{
                        width: '100%',
                        height: '20px',
                        background: '#ddd',
                        borderRadius: '10px',
                        overflow: 'hidden'
                    }}>
                        <div style={{
                            width: `${status.progress_percent}%`,
                            height: '100%',
                            background: '#4CAF50',
                            transition: 'width 0.3s'
                        }} />
                    </div>
                    <p>{status.message}</p>
                    {status.total_pages > 1 && (
                        <p>Page {status.current_page} of {status.total_pages}</p>
                    )}
                </div>
            )}

            {/* Results */}
            {results && (
                <div>
                    <h2>Results ({results.results.pages.length} page(s))</h2>
                    {results.results.pages.map((page) => (
                        <div key={page.page_number} style={{
                            border: '1px solid #ccc',
                            padding: '15px',
                            marginBottom: '15px',
                            borderRadius: '5px'
                        }}>
                            <h3>Page {page.page_number}</h3>

                            {/* Markdown Preview */}
                            <div style={{ marginBottom: '10px' }}>
                                <h4>Extracted Text:</h4>
                                <pre style={{
                                    background: '#f5f5f5',
                                    padding: '10px',
                                    borderRadius: '3px',
                                    whiteSpace: 'pre-wrap'
                                }}>
                                    {page.markdown.substring(0, 500)}
                                    {page.markdown.length > 500 && '...'}
                                </pre>
                            </div>

                            {/* Detected Elements */}
                            <div>
                                <h4>Detected Elements:</h4>
                                <p>
                                    {page.json_output.labels.length} elements found: {' '}
                                    {[...new Set(page.json_output.labels)].join(', ')}
                                </p>
                            </div>
                        </div>
                    ))}

                    {/* Download Button */}
                    <a
                        href={`${API_BASE}${results.download_url}`}
                        download
                        style={{
                            display: 'inline-block',
                            padding: '10px 20px',
                            background: '#2196F3',
                            color: 'white',
                            textDecoration: 'none',
                            borderRadius: '5px'
                        }}
                    >
                        Download ZIP
                    </a>
                </div>
            )}
        </div>
    );
}

export default DocumentOCR;
```

---

### React with WebSocket

```jsx
import React, { useState, useEffect } from 'react';

const API_BASE = 'https://isseygino911-dots-ocr-parser.hf.space';

function useWebSocketProgress(jobId) {
    const [status, setStatus] = useState(null);

    useEffect(() => {
        if (!jobId) return;

        const ws = new WebSocket(
            `wss://isseygino911-dots-ocr-parser.hf.space/api/jobs/${jobId}/stream`
        );

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            setStatus(message.data);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        return () => {
            ws.close();
        };
    }, [jobId]);

    return status;
}

// Usage in component
function MyComponent({ jobId }) {
    const status = useWebSocketProgress(jobId);

    if (!status) return <div>Loading...</div>;

    return (
        <div>
            <p>Status: {status.status}</p>
            <p>Progress: {status.progress_percent}%</p>
        </div>
    );
}
```

---

## Complete Workflows

### Workflow 1: Simple Upload and Download

```javascript
async function simpleWorkflow(file) {
    const API_BASE = 'https://isseygino911-dots-ocr-parser.hf.space';

    // 1. Upload
    const formData = new FormData();
    formData.append('file', file);

    const uploadResp = await fetch(`${API_BASE}/api/parse/image`, {
        method: 'POST',
        body: formData
    });
    const { job_id } = await uploadResp.json();

    // 2. Wait for completion
    let status;
    do {
        await new Promise(resolve => setTimeout(resolve, 2000));
        const statusResp = await fetch(`${API_BASE}/api/jobs/${job_id}/status`);
        status = await statusResp.json();
    } while (status.status === 'processing' || status.status === 'queued');

    // 3. Download results
    const downloadUrl = `${API_BASE}/api/jobs/${job_id}/download`;
    window.open(downloadUrl, '_blank');
}
```

---

### Workflow 2: Batch Processing

```javascript
async function processBatch(files) {
    const API_BASE = 'https://isseygino911-dots-ocr-parser.hf.space';
    const results = [];

    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);

        const uploadResp = await fetch(`${API_BASE}/api/parse/image`, {
            method: 'POST',
            body: formData
        });
        const { job_id } = await uploadResp.json();

        results.push({ filename: file.name, job_id });
    }

    // Monitor all jobs
    const completed = [];
    while (completed.length < results.length) {
        for (const job of results) {
            if (completed.includes(job.job_id)) continue;

            const statusResp = await fetch(`${API_BASE}/api/jobs/${job.job_id}/status`);
            const status = await statusResp.json();

            if (status.status === 'completed') {
                completed.push(job.job_id);
                console.log(`${job.filename} completed`);
            }
        }

        await new Promise(resolve => setTimeout(resolve, 2000));
    }

    return results;
}
```

---

## Error Handling

### Best Practices

```javascript
async function robustUpload(file) {
    const API_BASE = 'https://isseygino911-dots-ocr-parser.hf.space';
    const MAX_RETRIES = 3;
    const TIMEOUT = 300000; // 5 minutes

    try {
        // 1. Upload with retry
        let uploadResp;
        for (let i = 0; i < MAX_RETRIES; i++) {
            try {
                const formData = new FormData();
                formData.append('file', file);

                uploadResp = await fetch(`${API_BASE}/api/parse/image`, {
                    method: 'POST',
                    body: formData
                });

                if (uploadResp.ok) break;

                if (i === MAX_RETRIES - 1) {
                    throw new Error('Upload failed after retries');
                }

                await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
            } catch (error) {
                if (i === MAX_RETRIES - 1) throw error;
            }
        }

        const { job_id } = await uploadResp.json();

        // 2. Poll with timeout
        const startTime = Date.now();
        while (true) {
            if (Date.now() - startTime > TIMEOUT) {
                throw new Error('Processing timeout');
            }

            const statusResp = await fetch(`${API_BASE}/api/jobs/${job_id}/status`);

            if (!statusResp.ok) {
                throw new Error(`Status check failed: ${statusResp.status}`);
            }

            const status = await statusResp.json();

            if (status.status === 'completed') {
                return await fetch(`${API_BASE}/api/jobs/${job_id}/results`)
                    .then(r => r.json());
            }

            if (status.status === 'failed') {
                throw new Error(`Processing failed: ${status.error}`);
            }

            await new Promise(resolve => setTimeout(resolve, 2000));
        }

    } catch (error) {
        console.error('Error:', error);

        // Handle specific errors
        if (error.message.includes('timeout')) {
            alert('Processing is taking longer than expected. Please try again later.');
        } else if (error.message.includes('failed')) {
            alert('Processing failed. Please check your file and try again.');
        } else {
            alert('An error occurred. Please try again.');
        }

        throw error;
    }
}
```

---

## Performance Tips

1. **Use WebSocket for large PDFs** - More efficient than polling for documents with many pages
2. **Poll interval** - Use 2-3 seconds for polling to balance responsiveness and server load
3. **Timeout handling** - Set appropriate timeouts based on file size (60s for images, 300s for PDFs)
4. **Batch processing** - Process files sequentially to avoid overwhelming the server
5. **Error recovery** - Implement retry logic with exponential backoff

---

## Support

For issues or questions:
- API Documentation: `https://isseygino911-dots-ocr-parser.hf.space/docs`
- Test your integration at: [Interactive API Docs](https://isseygino911-dots-ocr-parser.hf.space/docs)

---

**Built with DotsOCR** - 1.7B parameter Vision-Language Model for Document Understanding
