# Complete RunPod Deployment Guide for DotsOCR

This guide will walk you through **every single step** to deploy your DotsOCR application on RunPod.ai.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Step 1: Build Docker Image](#step-1-build-docker-image)
3. [Step 2: Push to Docker Hub](#step-2-push-to-docker-hub)
4. [Step 3: Sign Up for RunPod](#step-3-sign-up-for-runpod)
5. [Step 4: Add Funds to RunPod](#step-4-add-funds-to-runpod)
6. [Step 5: Create RunPod Template](#step-5-create-runpod-template)
7. [Step 6: Deploy Your Pod](#step-6-deploy-your-pod)
8. [Step 7: Test Your Deployment](#step-7-test-your-deployment)
9. [Troubleshooting](#troubleshooting)
10. [Cost Estimation](#cost-estimation)

---

## Prerequisites

Before you start, make sure you have:

- ✅ Docker Desktop installed on your computer
  - **Mac**: Download from https://www.docker.com/products/docker-desktop/
  - **Windows**: Download from https://www.docker.com/products/docker-desktop/
  - **Linux**: Install via `sudo apt-get install docker.io` (Ubuntu/Debian)

- ✅ A Docker Hub account (free)
  - Sign up at https://hub.docker.com/signup

- ✅ A RunPod account (we'll create this in Step 3)

- ✅ A credit card for RunPod (they accept most major cards)

---

## Step 1: Build Docker Image

### 1.1 Open Terminal/Command Prompt

**Mac/Linux:**
- Press `Cmd + Space`, type "Terminal", press Enter

**Windows:**
- Press `Win + R`, type "cmd", press Enter

### 1.2 Navigate to Your Project

```bash
cd /path/to/dots_ocr
```

**Replace `/path/to/dots_ocr` with your actual project path.**

To find your path:
- **Mac**: Drag the folder into Terminal, it will show the path
- **Windows**: Open the folder in Explorer, click the address bar, copy the path

### 1.3 Verify Files Are Present

```bash
ls -la
```

You should see:
- ✅ `Dockerfile`
- ✅ `.dockerignore`
- ✅ `app.py`
- ✅ `requirements.txt`

### 1.4 Start Docker Desktop

**Mac/Windows:**
1. Open Docker Desktop application
2. Wait until you see "Docker Desktop is running" in the bottom left
3. The whale icon should be steady (not animated)

**Linux:**
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

### 1.5 Build the Docker Image

**IMPORTANT**: This step will take **30-60 minutes** because it downloads the large AI model (~10GB).

Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username:

```bash
docker build -t YOUR_DOCKERHUB_USERNAME/dots-ocr:latest .
```

**Example:**
```bash
docker build -t johnsmith/dots-ocr:latest .
```

**What you'll see:**
```
[+] Building 1234.5s (15/15) FINISHED
 => [internal] load build definition from Dockerfile
 => [internal] load .dockerignore
 => [stage-0  1/10] FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04
 ...
 => [stage-0  9/10] RUN python3 -c "from huggingface_hub..."
✅ Model pre-downloaded successfully
 => exporting to image
```

**☕ Go grab a coffee** - this will take a while!

### 1.6 Verify the Image Was Built

```bash
docker images
```

You should see your image:
```
REPOSITORY                          TAG       IMAGE ID       CREATED         SIZE
YOUR_USERNAME/dots-ocr             latest    abc123def456   2 minutes ago   15GB
```

---

## Step 2: Push to Docker Hub

### 2.1 Login to Docker Hub

```bash
docker login
```

**You'll be prompted:**
```
Username: [type your Docker Hub username]
Password: [type your Docker Hub password]
```

**Note:** When you type your password, you won't see any characters - this is normal!

**Success message:**
```
Login Succeeded
```

### 2.2 Push Your Image

**This will take 20-40 minutes** depending on your internet speed (uploading ~15GB).

```bash
docker push YOUR_DOCKERHUB_USERNAME/dots-ocr:latest
```

**Example:**
```bash
docker push johnsmith/dots-ocr:latest
```

**What you'll see:**
```
The push refers to repository [docker.io/johnsmith/dots-ocr]
abc123: Pushing [==>                ] 123.4MB/5.234GB
def456: Pushing [====>              ] 456.7MB/8.912GB
...
latest: digest: sha256:abc123... size: 4567
```

**Success message:**
```
latest: digest: sha256:abc123def456... size: 4567
```

### 2.3 Verify on Docker Hub

1. Go to https://hub.docker.com
2. Click "Sign In" (top right)
3. Enter your username and password
4. Click on your username (top right) → "Repositories"
5. You should see `dots-ocr` in the list
6. Click on it - you should see "latest" tag with size ~15GB

**✅ Your Docker image is now publicly available!**

---

## Step 3: Sign Up for RunPod

### 3.1 Create Account

1. Go to https://www.runpod.io
2. Click **"Sign Up"** (top right corner)
3. You have two options:

   **Option A: Sign up with GitHub (Recommended)**
   - Click "Continue with GitHub"
   - Click "Authorize RunPod" (if prompted)

   **Option B: Sign up with Email**
   - Enter your email address
   - Click "Continue"
   - Check your email for verification code
   - Enter the 6-digit code
   - Create a password

4. You'll be redirected to the RunPod dashboard

**✅ You should now see the RunPod Dashboard**

---

## Step 4: Add Funds to RunPod

RunPod uses a **prepaid credit system**. Minimum is $10.

### 4.1 Navigate to Billing

1. On the left sidebar, click **"Billing"** (dollar sign icon)
2. You'll see "Current Balance: $0.00"

### 4.2 Add Credits

1. Click the **"Add Credits"** button (purple/blue button)
2. Enter the amount (minimum $10, recommended $25 for testing)
   - $10 = ~4-5 hours on RTX 4090
   - $25 = ~10-12 hours on RTX 4090
3. Click **"Continue"**
4. Enter your credit card information:
   - Card number
   - Expiration date (MM/YY)
   - CVC (3-digit code on back)
   - Billing address
5. Click **"Add Credits"**

**Success:** You'll see "Credits added successfully" and your balance will update.

**✅ You now have credits to deploy pods!**

---

## Step 5: Create RunPod Template

Templates define how your container runs. This is the most important step!

### 5.1 Navigate to Templates

1. On the left sidebar, click **"Templates"** (looks like a document icon)
2. Click **"New Template"** button (top right)

### 5.2 Fill Out Template Form

**Template Name:**
```
DotsOCR Parser
```

**Template Type:**
- Select: **"Docker"** (should be selected by default)

**Container Image:**
```
YOUR_DOCKERHUB_USERNAME/dots-ocr:latest
```
**Example:** `johnsmith/dots-ocr:latest`

**Container Disk:**
```
50
```
(This is in GB - gives room for temporary files)

**Docker Command:**
- **Leave this BLANK** (we use CMD in Dockerfile)

**Expose HTTP Ports:**
```
7860
```
**IMPORTANT:** Click the **"+"** button next to the text box to add the port!

You should see: `7860/http` appear as a tag.

**Expose TCP Ports:**
- Leave blank

**Environment Variables:**

Click **"Add Environment Variable"** and add:

| Variable Name | Value |
|---------------|-------|
| `PORT` | `7860` |

**Volume Mount Path:**
```
/data
```

**Volume Size:**
```
50
```
(This is in GB - for uploads and results)

### 5.3 Advanced Settings (Optional but Recommended)

Scroll down to "Advanced" section:

**Container Start Command:**
- Leave blank (we use CMD in Dockerfile)

**Enable Public IP:**
- ✅ Check this box (allows external API access)

### 5.4 Save Template

1. Scroll to the bottom
2. Click **"Save Template"** (purple button)

**Success:** You'll see "Template created successfully"

**✅ Your template is now ready to use!**

---

## Step 6: Deploy Your Pod

Now we'll create an actual running instance (pod) from your template.

### 6.1 Navigate to Pods

1. On the left sidebar, click **"Pods"** (looks like a cube icon)
2. Click **"+ Deploy"** button (top right, or center if no pods exist)

### 6.2 Choose GPU Type

You'll see a list of available GPUs. **Recommended options:**

**Best for Testing (Cheapest):**
- **RTX 4090** - ~$0.40/hour
- Look for "Community Cloud" (cheaper than "Secure Cloud")
- Click on any available RTX 4090 pod

**Best for Production:**
- **RTX A6000** - ~$0.79/hour
- **A100 40GB** - ~$1.89/hour

**How to choose:**
1. Look at the "$/hr" column
2. Look at "Availability" - green dots mean available
3. Click **"Deploy"** button on your chosen GPU

### 6.3 Configure Pod

**Pod Name:**
```
dots-ocr-api
```
(You can name it anything)

**Select Template:**
1. Click the **"Select Template"** dropdown
2. Find and click **"DotsOCR Parser"** (the template you just created)
3. All settings will auto-fill from your template

**Verify Settings:**
- ✅ Container Image: `YOUR_USERNAME/dots-ocr:latest`
- ✅ Container Disk: 50 GB
- ✅ Volume Disk: 50 GB
- ✅ Expose HTTP Ports: 7860
- ✅ Environment Variables: PORT=7860

### 6.4 Deploy!

1. Scroll to bottom
2. Review the estimated cost per hour
3. Click **"Deploy On-Demand"** (purple button)

**You'll see:**
```
Creating pod...
```

**Wait 2-5 minutes** - the pod is starting up.

**Status will change:**
1. `Pending` → `Starting` → `Running`

---

## Step 7: Test Your Deployment

### 7.1 Get Your Pod URL

Once the pod status is **"Running"**:

1. In the Pods list, find your pod
2. You'll see a section called **"Connect"**
3. Click **"Connect to HTTP Service [7860]"**
4. This will show you the URL - it looks like:
   ```
   https://abc123-7860.proxy.runpod.net
   ```
5. **Copy this URL** - this is your API endpoint!

### 7.2 Test Health Endpoint

Open your browser and visit:
```
https://YOUR-POD-URL/health
```

**Example:**
```
https://abc123-7860.proxy.runpod.net/health
```

**You should see:**
```json
{
  "status": "healthy",
  "gpu_available": true
}
```

**✅ Your API is live!**

### 7.3 View API Documentation

Visit:
```
https://YOUR-POD-URL/docs
```

You'll see the interactive Swagger API documentation!

### 7.4 Test Image Upload

**Using cURL (Mac/Linux Terminal):**

```bash
curl -X POST "https://YOUR-POD-URL/api/parse/image" \
  -F "file=@/path/to/your/image.jpg" \
  -F "prompt_mode=prompt_layout_all_en"
```

**Using Python:**

```python
import requests
import time

# Your RunPod URL
BASE_URL = "https://abc123-7860.proxy.runpod.net"

# Upload image
with open("test_image.jpg", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/api/parse/image",
        files={"file": f},
        data={"prompt_mode": "prompt_layout_all_en"}
    )

job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")

# Poll for results
while True:
    status = requests.get(f"{BASE_URL}/api/jobs/{job_id}/status").json()
    print(f"Status: {status['status']} - {status['progress_percent']}%")

    if status['status'] == 'completed':
        print("✅ Done!")

        # Get results
        results = requests.get(f"{BASE_URL}/api/jobs/{job_id}/results").json()
        print(results['results']['pages'][0]['markdown'][:200])

        # Download ZIP
        zip_response = requests.get(f"{BASE_URL}/api/jobs/{job_id}/download")
        with open("results.zip", "wb") as f:
            f.write(zip_response.content)
        print("Downloaded results.zip")
        break

    time.sleep(2)
```

**✅ Your DotsOCR API is fully deployed on RunPod!**

---

## Troubleshooting

### Problem: Pod Won't Start

**Symptoms:** Pod status stays "Pending" or "Failed"

**Solutions:**
1. Check if you have enough credits (Billing → Current Balance)
2. Try a different GPU type - some may be out of stock
3. Check the pod logs:
   - Click on the pod name
   - Click "Logs" tab
   - Look for error messages

### Problem: Can't Access URL

**Symptoms:** URL returns "Connection refused" or timeout

**Solutions:**
1. Make sure pod status is "Running" (not "Starting")
2. Wait 2-3 minutes after pod starts (model is loading)
3. Check the pod logs for startup errors
4. Verify port 7860 is exposed (Edit pod → Expose Ports)

### Problem: "CUDA Not Available" Error

**Symptoms:** API returns error about CUDA/GPU

**Solutions:**
1. Make sure you selected a GPU pod (not CPU-only)
2. Restart the pod:
   - Click pod name
   - Click "Stop"
   - Wait 10 seconds
   - Click "Start"

### Problem: "Model Not Found" Error

**Symptoms:** API crashes with HuggingFace download errors

**Solutions:**
1. The model wasn't pre-downloaded during Docker build
2. Rebuild your Docker image (Step 1) and ensure you see:
   ```
   ✅ Model pre-downloaded successfully
   ```
3. Push the new image to Docker Hub
4. Recreate the pod with the new image

### Problem: Slow Performance

**Symptoms:** Processing takes >30 seconds per page

**Solutions:**
1. Upgrade to a better GPU:
   - RTX 4090 is good for testing
   - A100 is better for production
2. Check pod utilization:
   - Click pod name → Metrics tab
   - GPU should be near 100% during inference

---

## Cost Estimation

### GPU Pricing (as of 2024)

| GPU Type | $/hour | Good For |
|----------|--------|----------|
| **RTX 3090** | $0.34 | Testing, low volume |
| **RTX 4090** | $0.39 | Testing, medium volume |
| **RTX A6000** | $0.79 | Production, high accuracy |
| **A100 40GB** | $1.89 | Production, fast processing |
| **A100 80GB** | $2.89 | Production, batch processing |

**Note:** Prices vary by availability and cloud type (Community vs Secure).

### Usage Examples

**Light Testing (10 hours/month):**
- GPU: RTX 4090 ($0.39/hr)
- Cost: $3.90/month

**Medium Production (100 hours/month):**
- GPU: RTX A6000 ($0.79/hr)
- Cost: $79/month

**Heavy Production (24/7):**
- GPU: RTX A6000 ($0.79/hr)
- Cost: ~$570/month
- Consider: A100 for better performance

### Processing Speed

| GPU | Pages/minute | Cost per 100 pages |
|-----|-------------|-------------------|
| RTX 3090 | ~4 | $0.51 |
| RTX 4090 | ~5 | $0.47 |
| RTX A6000 | ~6 | $0.79 |
| A100 40GB | ~10 | $1.13 |

---

## Managing Your Pod

### To Stop Your Pod (Save Money)

1. Go to "Pods" in the left sidebar
2. Find your pod
3. Click the **"Stop"** button (square icon)
4. Confirm

**Note:** You're only charged when the pod is Running!

### To Start Your Pod Again

1. Find the stopped pod
2. Click the **"Start"** button (play icon)
3. Wait 2-3 minutes for it to start

**Your API URL will remain the same!**

### To Delete Your Pod

1. Stop the pod first
2. Click the **"Trash"** icon
3. Confirm deletion

**Warning:** This deletes all data on the pod's volume!

---

## Next Steps

**✅ You're all set!** Your DotsOCR API is running on RunPod.

**Recommended:**

1. **Integrate with your frontend:**
   - Use the RunPod URL in your React app
   - Update CORS settings if needed

2. **Set up monitoring:**
   - Check pod metrics regularly (Metrics tab)
   - Monitor credit usage (Billing page)

3. **Add API authentication:**
   - Add an API key system to `app.py`
   - Prevent unauthorized usage

4. **Consider autoscaling:**
   - Use RunPod Serverless for variable workloads
   - Only pay when processing documents

5. **Backup strategy:**
   - Since data is temporary, ensure users download results immediately
   - Consider adding S3 upload after processing

---

## Support

**RunPod Discord:**
https://discord.gg/runpod

**RunPod Documentation:**
https://docs.runpod.io

**DotsOCR Issues:**
Create an issue in your GitHub repo

---

**Congratulations! 🎉**

You've successfully deployed DotsOCR on RunPod.ai!
