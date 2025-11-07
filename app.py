import subprocess
import sys
import os

print("=" * 50)
print("Installing flash-attn (needs torch first)...")
print("=" * 50)

# Install flash-attn now that torch is available from requirements.txt
subprocess.run([
    sys.executable, "-m", "pip", "install",
    "flash-attn==2.8.0.post2", "--no-build-isolation"
], check=True)

print("=" * 50)
print("Cloning dots.ocr repository...")
print("=" * 50)

# Clone dots.ocr if not exists
if not os.path.exists("dots.ocr"):
    subprocess.run([
        "git", "clone", 
        "https://github.com/rednote-hilab/dots.ocr.git"
    ], check=True)

print("=" * 50)
print("Installing dots.ocr...")
print("=" * 50)

# Install dots.ocr without dependencies (we already have them)
subprocess.run([
    sys.executable, "-m", "pip", "install", 
    "-e", "./dots.ocr", "--no-deps"
], check=True)

print("=" * 50)
print("Starting Gradio interface...")
print("=" * 50)

# Run the demo
os.chdir("dots.ocr")
subprocess.run([
    sys.executable, "demo/demo_gradio.py",
    "--server_name", "0.0.0.0",
    "--server_port", "7860"
])
