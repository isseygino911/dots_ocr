import subprocess
import sys
import os

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn==2.8.0.post2", "--no-build-isolation"], check=True)

if not os.path.exists("dots.ocr"):
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr", "--no-deps"], check=True)

# Fix the module path issue by downloading to a path without periods
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import shutil

print("Downloading model to local path...")
# Download to a local directory without periods
local_model_path = "./DotsOCR_model"

if not os.path.exists(local_model_path):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="rednote-hilab/dots.ocr",
        local_dir=local_model_path,
        local_dir_use_symlinks=False
    )

print("Loading model from local path...")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True, local_files_only=True)
print("Model loaded!")

def parse_image(image):
    if image is None:
        return "Please upload an image", ""
    
    prompt = """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.
1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
3. Text Extraction & Formatting Rules:
- Picture: omit text field
- Formula: LaTeX format
- Table: HTML format
- All Others: Markdown format
4. Constraints:
- Original text only, no translation
- Sort by reading order
5. Final Output: Single JSON object"""
    
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8000)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    try:
        result = json.loads(output_text)
        markdown_parts = []
        for item in result.get("layout", []):
            if "text" in item and item.get("category") not in ["Picture", "Page-header", "Page-footer"]:
                markdown_parts.append(item["text"])
        markdown_text = "\n\n".join(markdown_parts)
        return markdown_text, json.dumps(result, indent=2, ensure_ascii=False)
    except:
        return output_text, output_text

demo = gr.Interface(
    fn=parse_image,
    inputs=gr.Image(type="pil", label="Upload Document Image"),
    outputs=[
        gr.Textbox(label="Extracted Text", lines=15),
        gr.Code(label="JSON Output", language="json", lines=15)
    ],
    title="dots.ocr Document Parser",
    description="Upload a document image to extract text, tables, and formulas. Supports 100+ languages!"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
