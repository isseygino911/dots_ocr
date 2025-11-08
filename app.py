import subprocess
import sys
import os

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn==2.8.0.post2", "--no-build-isolation"], check=True)

if not os.path.exists("dots.ocr"):
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr", "--no-deps"], check=True)

# Download model to the path their code expects: ./weights/DotsOCR
print("Downloading model to ./weights/DotsOCR...")
os.makedirs("./weights", exist_ok=True)
model_path = "./weights/DotsOCR"

if not os.path.exists(model_path):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="rednote-hilab/dots.ocr",
        local_dir=model_path,
        local_dir_use_symlinks=False
    )

print("Model ready. Starting demo...")

# Add dots.ocr to path
sys.path.insert(0, os.path.abspath("./dots.ocr"))

# Change to dots.ocr directory
os.chdir("./dots.ocr")

# Modify demo to use HF backend
demo_file = "demo/demo_gradio.py"
with open(demo_file, 'r') as f:
    demo_code = f.read()

# Add use_hf=True to DotsOCRParser instantiation
demo_code = demo_code.replace(
    '''dots_parser = DotsOCRParser(
    ip=DEFAULT_CONFIG['ip'],
    port=DEFAULT_CONFIG['port_vllm'],
    dpi=200,
    min_pixels=DEFAULT_CONFIG['min_pixels'],
    max_pixels=DEFAULT_CONFIG['max_pixels']
)''',
    '''dots_parser = DotsOCRParser(
    ip=DEFAULT_CONFIG['ip'],
    port=DEFAULT_CONFIG['port_vllm'],
    dpi=200,
    min_pixels=DEFAULT_CONFIG['min_pixels'],
    max_pixels=DEFAULT_CONFIG['max_pixels'],
    use_hf=True
)'''
)

# Set port argument
sys.argv = ['demo_gradio.py', '7860']

# Execute
print("Launching Gradio demo with HF Transformers backend...")
exec(demo_code)
