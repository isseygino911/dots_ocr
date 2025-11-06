import gradio as gr
import subprocess
import sys
import os

# Install dots.ocr on first run
if not os.path.exists("dots.ocr"):
    print("Cloning dots.ocr...")
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)
    print("Installing dots.ocr...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr"], check=True)

# Change to dots.ocr directory and run their demo
os.chdir("dots.ocr")
subprocess.run([sys.executable, "demo/demo_gradio.py"])
