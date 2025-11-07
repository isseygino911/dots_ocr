import subprocess
import sys
import os

print("=" * 50)
print("Installing flash-attn (needs torch first)...")
print("=" * 50)

# Install flash-attn now that torch is available
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

# Install dots.ocr without dependencies
subprocess.run([
    sys.executable, "-m", "pip", "install", 
    "-e", "./dots.ocr", "--no-deps"
], check=True)

print("=" * 50)
print("Starting Gradio interface...")
print("=" * 50)

# Run the demo - their script doesn't support those arguments
os.chdir("dots.ocr")

# Just run it with the port as the first argument
subprocess.run([sys.executable, "demo/demo_gradio.py"])
