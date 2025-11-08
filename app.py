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
    print("Model downloaded!")

# Set environment variable to use local model
os.environ['HF_MODEL_PATH'] = os.path.abspath(local_model_path)

# Change to dots.ocr directory
os.chdir("dots.ocr")

# Modify demo_gradio.py to use local model path instead of HuggingFace Hub
print("Patching demo_gradio.py to use local model...")
with open("demo/demo_gradio.py", "r") as f:
    demo_code = f.read()

# Replace model path
demo_code = demo_code.replace(
    'model_path = "rednote-hilab/dots.ocr"',
    f'model_path = "{os.path.abspath(local_model_path)}"'
)

# Add local_files_only=True
demo_code = demo_code.replace(
    'trust_remote_code=True',
    'trust_remote_code=True, local_files_only=True'
)

# Fix launch to work with HF Spaces
demo_code = demo_code.replace(
    'demo.launch()',
    'demo.launch(server_name="0.0.0.0", server_port=7860, share=False)'
)

# Write patched version
with open("demo/demo_gradio_patched.py", "w") as f:
    f.write(demo_code)

print("Starting original demo_gradio.py...")
# Run the patched demo
subprocess.run([sys.executable, "demo/demo_gradio_patched.py"])
