import subprocess
import sys
import os

# Clone dots.ocr if not exists
if not os.path.exists("dots.ocr"):
    print("Cloning dots.ocr repository...")
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

# Install dots.ocr (dependencies already in requirements.txt)
print("Installing dots.ocr...")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr", "--no-deps"], check=True)

# Run the demo
print("Starting Gradio interface...")
os.chdir("dots.ocr")
subprocess.run([sys.executable, "demo/demo_gradio.py", "--server_name", "0.0.0.0", "--server_port", "7860"])
