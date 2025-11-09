#!/usr/bin/env python3
import subprocess
import sys
import os

print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "flash-attn==2.8.0.post2", "--no-build-isolation"], check=True)

if not os.path.exists("dots.ocr"):
    print("Cloning dots.ocr repository...")
    subprocess.run(["git", "clone", "https://github.com/rednote-hilab/dots.ocr.git"], check=True)

print("Installing dots.ocr...")
subprocess.run([sys.executable, "-m", "pip", "install", "-e", "./dots.ocr"], check=True)

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

os.chdir("./dots.ocr")

print("Patching demo for HF backend...")
sys.path.insert(0, os.getcwd())

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

# Reorganize layout: Preview on top, results below
# Find the main layout section with gr.Row()
old_layout = '''            with gr.Column(scale=6, variant="compact"):
                with gr.Row():
                    # Result Image
                    with gr.Column(scale=3):
                        gr.Markdown("### 👁️ File Preview")
                        result_image = gr.Image(
                            label="Layout Preview",
                            visible=True,
                            height=800,
                            show_label=False
                        )
                        
                        # Page navigation (shown during PDF preview)
                        with gr.Row():
                            prev_btn = gr.Button("⬅ Previous", size="sm")
                            page_info = gr.HTML(
                                value="<div id='page_info_box'>0 / 0</div>", 
                                elem_id="page_info_html"
                            )
                            next_btn = gr.Button("Next ➡", size="sm")
                        
                        # Info Display
                        info_display = gr.Markdown(
                            "Waiting for processing results...",
                            elem_id="info_box"
                        )
                    
                    # Markdown Result
                    with gr.Column(scale=3):
                        gr.Markdown("### ✔️ Result Display")
                        
                        with gr.Tabs(elem_id="markdown_tabs"):
                            with gr.TabItem("Markdown Render Preview"):
                                md_output = gr.Markdown(
                                    "## Please click the parse button to parse or select for single-task recognition...",
                                    
                                    latex_delimiters=[
                                        {"left": "$$", "right": "$$", "display": True},
                                        {"left": "$", "right": "$", "display": False}
                                    ],
                                    
                                    elem_id="markdown_output"
                                )
                            
                            with gr.TabItem("Markdown Raw Text"):
                                md_raw_output = gr.Textbox(
                                    value="🕐 Waiting for parsing result...",
                                    label="Markdown Raw Text",
                                    max_lines=100,
                                    lines=38,
                                    
                                    elem_id="markdown_output",
                                    show_label=False
                                )
                            
                            with gr.TabItem("Current Page JSON"):
                                current_page_json = gr.Textbox(
                                    value="🕐 Waiting for parsing result...",
                                    label="Current Page JSON",
                                    max_lines=100,
                                    lines=38,
                                    
                                    elem_id="markdown_output",
                                    show_label=False
                                )'''

new_layout = '''            with gr.Column(scale=6, variant="compact"):
                # Preview on top
                gr.Markdown("### 👁️ File Preview")
                result_image = gr.Image(
                    label="Layout Preview",
                    visible=True,
                    height=500,
                    show_label=False
                )
                
                # Page navigation
                with gr.Row():
                    prev_btn = gr.Button("⬅ Previous", size="sm")
                    page_info = gr.HTML(
                        value="<div id='page_info_box'>0 / 0</div>", 
                        elem_id="page_info_html"
                    )
                    next_btn = gr.Button("Next ➡", size="sm")
                
                # Info Display
                info_display = gr.Markdown(
                    "Waiting for processing results...",
                    elem_id="info_box"
                )
                
                # Results below
                gr.Markdown("### ✔️ Result Display")
                
                with gr.Tabs(elem_id="markdown_tabs"):
                    with gr.TabItem("Markdown Render Preview"):
                        md_output = gr.Markdown(
                            "## Please click the parse button to parse or select for single-task recognition...",
                            latex_delimiters=[
                                {"left": "$$", "right": "$$", "display": True},
                                {"left": "$", "right": "$", "display": False}
                            ],
                            elem_id="markdown_output"
                        )
                    
                    with gr.TabItem("Markdown Raw Text"):
                        md_raw_output = gr.Textbox(
                            value="🕐 Waiting for parsing result...",
                            label="Markdown Raw Text",
                            max_lines=100,
                            lines=20,
                            elem_id="markdown_output",
                            show_label=False
                        )
                    
                    with gr.TabItem("Current Page JSON"):
                        current_page_json = gr.Textbox(
                            value="🕐 Waiting for parsing result...",
                            label="Current Page JSON",
                            max_lines=100,
                            lines=20,
                            elem_id="markdown_output",
                            show_label=False
                        )'''

demo_code = demo_code.replace(old_layout, new_layout)

# Add CSS fixes
css_addition = '''
    
    /* Fix overflow and scrolling */
    #markdown_output {
        overflow: auto;
        max-width: 100%;
    }
    
    /* Ensure preview always shows */
    #result_image {
        display: block !important;
    }
'''

demo_code = demo_code.replace('    footer {', css_addition + '\n    footer {')

# Suppress console warnings by adding to the end
demo_code = demo_code.replace(
    'demo.launch(server_name="0.0.0.0", server_port=port, debug=True)',
    'demo.launch(server_name="0.0.0.0", server_port=port, show_api=False)'
)

sys.argv = ['demo_gradio.py', '7860']
print("Starting Gradio demo on port 7860...")
exec(demo_code)
