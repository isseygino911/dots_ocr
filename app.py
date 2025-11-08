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

# Now patch their demo to work with HF Transformers instead of vLLM
sys.path.insert(0, "./dots.ocr")
os.chdir("./dots.ocr")

# Modify the DotsOCRParser to use HuggingFace backend by default
import dots_ocr.parser as parser_module

# Monkey-patch to use HuggingFace backend
original_init = parser_module.DotsOCRParser.__init__

def patched_init(self, *args, **kwargs):
    # Force use_hf=True for HuggingFace Spaces
    kwargs['use_hf'] = True
    kwargs['model_path'] = local_model_path
    return original_init(self, *args, **kwargs)

parser_module.DotsOCRParser.__init__ = patched_init

# Now run their demo, but pass port 7860
sys.argv = ['demo_gradio.py', '7860']

# Execute their demo
exec(open('demo/demo_gradio.py').read())
