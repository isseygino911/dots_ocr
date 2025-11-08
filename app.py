import subprocess
import sys
import os

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn==2.8.0.post2", "--no-build-isolation"], check=True)

if not os.path.exists("dots.ocr"):
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr", "--no-deps"], check=True)

# Download model to ABSOLUTE path at ./weights/DotsOCR
print("Downloading model to ./weights/DotsOCR...")
os.makedirs("./weights", exist_ok=True)
model_path = os.path.abspath("./weights/DotsOCR")

if not os.path.exists(model_path):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="rednote-hilab/dots.ocr",
        local_dir=model_path,
        local_dir_use_symlinks=False
    )

print(f"Model downloaded to: {model_path}")

# Patch the parser to use absolute path
sys.path.insert(0, os.path.abspath("./dots.ocr"))

import dots_ocr.parser as parser_module

# Monkey-patch the hardcoded path to use absolute path
original_load = parser_module.DotsOCRParser._load_hf_model

def patched_load(self):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    # Use ABSOLUTE path instead of relative
    model_path_abs = os.path.abspath("./weights/DotsOCR")
    print(f"Loading model from: {model_path_abs}")
    
    self.model = AutoModelForCausalLM.from_pretrained(
        model_path_abs,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )
    self.processor = AutoProcessor.from_pretrained(
        model_path_abs, 
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True
    )
    self.process_vision_info = process_vision_info

parser_module.DotsOCRParser._load_hf_model = patched_load

# Change to dots.ocr directory and run demo
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
