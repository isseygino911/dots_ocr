#!/usr/bin/env python3
import subprocess
import sys
import os
import time

print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn==2.8.0.post2", "--no-build-isolation"], check=True)

# Clone their repo
if not os.path.exists("dots.ocr"):
    print("Cloning dots.ocr repository...")
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

# Install dots.ocr
print("Installing dots.ocr...")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr"], check=True)

# Download model to weights/DotsOCR (their expected location)
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

# Change to their directory
os.chdir("./dots.ocr")

# Patch their demo to use HF backend with correct path
print("Patching demo for HF backend...")
sys.path.insert(0, os.getcwd())

# Patch the parser module before it loads
import dots_ocr.parser as parser_module

original_load = parser_module.DotsOCRParser._load_hf_model

def patched_load(self):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    model_path = os.path.abspath("./weights/DotsOCR")
    print(f"Loading model from: {model_path}")
    
    self.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    self.processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    self.process_vision_info = process_vision_info

parser_module.DotsOCRParser._load_hf_model = patched_load

# Modify demo file to enable HF backend and fix Gradio issues
with open("demo/demo_gradio.py", 'r') as f:
    demo_code = f.read()

# Enable HF backend
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

# Fix Gradio 4.44.0 compatibility
demo_code = demo_code.replace('max_height=600,', '')
demo_code = demo_code.replace('show_copy_button=True,', '')
demo_code = demo_code.replace('show_copy_button=False,', '')
demo_code = demo_code.replace('theme="ocean"', 'theme=gr.themes.Soft()')

# Run the demo
sys.argv = ['demo_gradio.py', '7860']
print("Starting Gradio demo on port 7860...")
exec(demo_code)
