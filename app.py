import subprocess
import sys
import os

# Install flash-attn
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "flash-attn==2.8.0.post2", "--no-build-isolation"
], check=True)

# Clone and install dots.ocr
if not os.path.exists("dots.ocr"):
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr", "--no-deps"], check=True)

# Add to Python path
sys.path.insert(0, "./dots.ocr")
os.chdir("./dots.ocr")

# Import and run their demo with proper server settings
import demo.demo_gradio as demo_module

# Monkey-patch the launch call to work with HF Spaces
import gradio as gr
original_launch = gr.Blocks.launch

def patched_launch(self, *args, **kwargs):
    # Override with HF Spaces compatible settings
    kwargs['server_name'] = '0.0.0.0'
    kwargs['server_port'] = 7860
    kwargs['share'] = False
    return original_launch(self, *args, **kwargs)

gr.Blocks.launch = patched_launch

# Now run the demo - it will use our patched launch
exec(open("demo/demo_gradio.py 7890").read())
