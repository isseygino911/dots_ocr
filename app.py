import subprocess
import sys
import os

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn==2.8.0.post2", "--no-build-isolation"], check=True)

if not os.path.exists("dots.ocr"):
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr", "--no-deps"], check=True)

print("Downloading model...")
local_model_path = "./DotsOCR_model"

if not os.path.exists(local_model_path):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="rednote-hilab/dots.ocr",
        local_dir=local_model_path,
        local_dir_use_symlinks=False
    )

print("Setting up environment...")

# Set environment variable for the model path
os.environ['HF_MODEL_PATH'] = os.path.abspath(local_model_path)

# Add dots.ocr to path
sys.path.insert(0, os.path.abspath("./dots.ocr"))

# Modify the demo script to use HF backend
demo_file = "./dots.ocr/demo/demo_gradio.py"

with open(demo_file, 'r') as f:
    demo_code = f.read()

# Replace the DotsOCRParser instantiation to use HF backend
demo_code = demo_code.replace(
    "dots_parser = DotsOCRParser(",
    "dots_parser = DotsOCRParser(use_hf=True, "
)

# Change to dots.ocr directory
os.chdir("./dots.ocr")

# Set port argument
sys.argv = ['demo_gradio.py', '7860']

# Execute modified demo
print("Starting Gradio demo with HF Transformers backend...")
exec(demo_code)
