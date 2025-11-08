import subprocess
import sys
import os

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn==2.8.0.post2", "--no-build-isolation"], check=True)

if not os.path.exists("dots.ocr"):
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr", "--no-deps"], check=True)

print("Downloading model to path without periods...")
local_model_path = "./DotsOCR_model"

if not os.path.exists(local_model_path):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="rednote-hilab/dots.ocr",
        local_dir=local_model_path,
        local_dir_use_symlinks=False
    )

print("Model downloaded. Starting Gradio with HF Transformers backend...")

# Add dots.ocr to Python path BEFORE importing
sys.path.insert(0, os.path.abspath("./dots.ocr"))

# Now import and patch
from dots_ocr.parser import DotsOCRParser

# Monkey-patch to use HuggingFace backend
original_init = DotsOCRParser.__init__

def patched_init(self, ip=None, port=None, *args, **kwargs):
    # Force use_hf=True for HuggingFace Spaces
    kwargs['use_hf'] = True
    kwargs['model_path'] = local_model_path
    # Don't need ip/port for HF backend
    return original_init(self, ip=ip or "127.0.0.1", port=port or 8000, *args, **kwargs)

DotsOCRParser.__init__ = patched_init

# Change to dots.ocr directory and run their demo
os.chdir("./dots.ocr")

# Set command-line arguments for their demo
sys.argv = ['demo_gradio.py', '7860']

# Execute their demo
with open('demo/demo_gradio.py', 'r') as f:
    demo_code = f.read()
    exec(demo_code)
