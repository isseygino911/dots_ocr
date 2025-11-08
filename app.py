import subprocess
import sys
import os

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn==2.8.0.post2", "--no-build-isolation"], check=True)

if not os.path.exists("dots.ocr"):
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr", "--no-deps"], check=True)

# Download model to /home/user/app/weights/DotsOCR (BEFORE cd to dots.ocr)
print("Downloading model...")
weights_dir = os.path.abspath("./weights")
os.makedirs(weights_dir, exist_ok=True)
model_path = os.path.join(weights_dir, "DotsOCR")

if not os.path.exists(model_path):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="rednote-hilab/dots.ocr",
        local_dir=model_path,
        local_dir_use_symlinks=False
    )

print(f"Model ready at: {model_path}")

# Add dots.ocr to path
sys.path.insert(0, os.path.abspath("./dots.ocr"))

# Patch parser BEFORE importing/executing
import dots_ocr.parser as parser_module

# Store the absolute model path for the patch
ABSOLUTE_MODEL_PATH = model_path

original_load = parser_module.DotsOCRParser._load_hf_model

def patched_load(self):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    print(f"Loading HF model from: {ABSOLUTE_MODEL_PATH}")
    
    self.model = AutoModelForCausalLM.from_pretrained(
        ABSOLUTE_MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    self.processor = AutoProcessor.from_pretrained(
        ABSOLUTE_MODEL_PATH,
        trust_remote_code=True,
        use_fast=True
    )
    self.process_vision_info = process_vision_info

parser_module.DotsOCRParser._load_hf_model = patched_load

# NOW change to dots.ocr directory
os.chdir("./dots.ocr")

# Modify demo to use HF backend
demo_file = "demo/demo_gradio.py"
with open(demo_file, 'r') as f:
    demo_code = f.read()

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

sys.argv = ['demo_gradio.py', '7860']

print("Launching Gradio demo...")
exec(demo_code)
