import subprocess
import sys
import os

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn==2.8.0.post2", "--no-build-isolation"], check=True)

if not os.path.exists("dots.ocr"):
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr", "--no-deps"], check=True)

# Now create enhanced Gradio interface with ALL features
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import json
import fitz  # PyMuPDF
import io

print("Loading model...")
local_model_path = "./DotsOCR_model"

if not os.path.exists(local_model_path):
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="rednote-hilab/dots.ocr",
        local_dir=local_model_path,
        local_dir_use_symlinks=False
    )

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

# Different prompt modes from dots.ocr
PROMPTS = {
    "Parse All (Detection + Recognition)": """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.
1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
3. Text Extraction & Formatting Rules:
- Picture: omit text field
- Formula: LaTeX format
- Table: HTML format
- All Others: Markdown format
4. Constraints:
- Original text only, no translation
- All layout elements must be sorted according to human reading order
5. Final Output: Single JSON object""",
    
    "Layout Detection Only": """Please output the layout detection results from the PDF image, including each layout element's bbox and its category.
1. Bbox format: [x1, y1, x2, y2]
2. Layout Categories: ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']
3. Constraints:
- All layout elements must be sorted according to human reading order
4. Final Output: Single JSON object""",
    
    "OCR Only (No Headers/Footers)": """Please output the text content from the PDF image, excluding page headers and footers.
Format the output as Markdown.""",
}

def pdf_to_images(pdf_file, dpi=200):
    """Convert PDF to images"""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images

def parse_document(file, prompt_mode, bbox_coords=None, dpi=200):
    """Parse document with selected mode"""
    if file is None:
        return "Please upload an image or PDF", "", ""
    
    # Handle PDF or Image
    if file.name.lower().endswith('.pdf'):
        images = pdf_to_images(file, dpi=dpi)
        if len(images) == 0:
            return "Failed to extract images from PDF", "", ""
        # For demo, process first page only
        image = images[0]
        status = f"📄 Processing page 1 of {len(images)} (showing first page only)"
    else:
        image = Image.open(file)
        status = "🖼️ Processing image"
    
    # Get prompt based on mode
    prompt = PROMPTS[prompt_mode]
    
    # Build messages
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    
    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=12000, temperature=0.1)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    # Parse results based on mode
    try:
        if "JSON" in prompt:
            result = json.loads(output_text)
            json_output = json.dumps(result, indent=2, ensure_ascii=False)
            
            # Extract markdown (skip headers/footers)
            markdown_parts = []
            for item in result.get("layout", []):
                if "text" in item:
                    category = item.get("category", "")
                    if category not in ["Picture", "Page-header", "Page-footer"]:
                        text_content = item["text"]
                        # Add category as header for clarity
                        if category in ["Title", "Section-header"]:
                            markdown_parts.append(f"## {text_content}")
                        elif category == "Formula":
                            markdown_parts.append(f"$$\n{text_content}\n$$")
                        else:
                            markdown_parts.append(text_content)
            
            markdown_text = "\n\n".join(markdown_parts)
            return status + " ✅", markdown_text, json_output
        else:
            # OCR only mode
            return status + " ✅", output_text, ""
    except Exception as e:
        return f"{status} ⚠️ Parse error: {str(e)}", output_text, output_text

# Create enhanced interface
with gr.Blocks(title="dots.ocr Document Parser", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎯 dots.ocr - Multilingual Document Parser
    
    **State-of-the-art document parsing with 1.7B parameters**
    
    Upload a document image or PDF to extract:
    - 📝 **Text** with proper reading order
    - 📊 **Tables** in HTML format
    - 🔢 **Formulas** in LaTeX
    - 📐 **Layout** detection (bounding boxes)
    - 🌍 **100+ languages** supported
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="📁 Upload Document",
                file_types=[".jpg", ".jpeg", ".png", ".pdf"],
                type="filepath"
            )
            
            prompt_dropdown = gr.Dropdown(
                choices=list(PROMPTS.keys()),
                value="Parse All (Detection + Recognition)",
                label="🎨 Parsing Mode",
                info="Choose what to extract"
            )
            
            with gr.Accordion("⚙️ Advanced Options", open=False):
                dpi_slider = gr.Slider(
                    minimum=100,
                    maximum=300,
                    value=200,
                    step=50,
                    label="PDF DPI (higher = better quality, slower)",
                    info="Recommended: 200"
                )
            
            parse_btn = gr.Button("🚀 Parse Document", variant="primary", size="lg")
            
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=1
            )
    
    with gr.Row():
        with gr.Column(scale=1):
            markdown_output = gr.Textbox(
                label="📝 Extracted Text (Markdown)",
                lines=20,
                max_lines=40
            )
        
        with gr.Column(scale=1):
            json_output = gr.Code(
                label="📋 Full JSON Output",
                language="json",
                lines=20
            )
    
    # Parse action
    parse_btn.click(
        fn=parse_document,
        inputs=[file_input, prompt_dropdown, gr.State(None), dpi_slider],
        outputs=[status_box, markdown_output, json_output]
    )
    
    gr.Markdown("""
    ---
    ### 📚 Features
    
    **Parsing Modes:**
    - **Parse All**: Full layout detection + content extraction with bounding boxes
    - **Layout Detection Only**: Detect regions without extracting text
    - **OCR Only**: Extract text without headers/footers
    
    **Supported Formats:**
    - Images: JPG, PNG, JPEG
    - Documents: PDF (processes first page in demo)
    
    **Output Formats:**
    - 📝 Markdown for text
    - 📊 HTML for tables
    - 🔢 LaTeX for formulas
    - 📋 JSON for structured data
    
    ### 💡 Tips
    - For best results, use images with DPI ~200
    - Optimal resolution: under 11.3 megapixels
    - Complex tables may need manual verification
    - Formulas are output in LaTeX format
    
    ### 🔗 Links
    - **Model**: [rednote-hilab/dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr)
    - **GitHub**: [dots.ocr](https://github.com/rednote-hilab/dots.ocr)
    - **Paper**: Coming soon
    """)

demo.launch(server_name="0.0.0.0", server_port=7860)
