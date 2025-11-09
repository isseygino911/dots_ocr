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

print("Patching demo with progress tracking...")
sys.path.insert(0, os.getcwd())

import dots_ocr.parser as parser_module

original_load = parser_module.DotsOCRParser._load_hf_model

def patched_load(self):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from qwen_vl_utils import process_vision_info
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required but not available!")
    
    model_path = os.path.abspath("./weights/DotsOCR")
    print(f"Loading model from: {model_path}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    self.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    
    self.model.eval()
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    self.processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    self.process_vision_info = process_vision_info
    
    print(f"✅ Model loaded")

parser_module.DotsOCRParser._load_hf_model = patched_load

# Optimize inference
original_inference = parser_module.DotsOCRParser._inference_with_hf

def optimized_inference(self, image, prompt):
    import torch
    import time
    
    start = time.time()
    print(f"🔄 Starting inference...")
    
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]

    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = self.process_vision_info(messages)
    inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")

    print(f"⏱️  Preprocessing: {time.time()-start:.1f}s")
    gen_start = time.time()

    with torch.inference_mode():
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=6000,  # Further reduced
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    
    print(f"⏱️  Generation: {time.time()-gen_start:.1f}s")
        
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    torch.cuda.empty_cache()
    
    print(f"✅ Total: {time.time()-start:.1f}s")
    
    return response

parser_module.DotsOCRParser._inference_with_hf = optimized_inference

# Patch PDF processing to show progress
original_parse_pdf = parser_module.DotsOCRParser.parse_pdf

def parse_pdf_with_progress(self, input_path, filename, prompt_mode, save_dir):
    import time
    print(f"\n{'='*60}")
    print(f"📄 Starting PDF processing: {input_path}")
    print(f"{'='*60}\n")
    
    from dots_ocr.utils.doc_utils import load_images_from_pdf
    
    total_start = time.time()
    print(f"📖 Loading PDF pages...")
    images_origin = load_images_from_pdf(input_path, dpi=self.dpi)
    total_pages = len(images_origin)
    print(f"✅ Loaded {total_pages} pages in {time.time()-total_start:.1f}s\n")
    
    results = []
    for i, image in enumerate(images_origin):
        page_start = time.time()
        print(f"\n{'─'*60}")
        print(f"📄 Processing page {i+1}/{total_pages}...")
        print(f"{'─'*60}")
        
        result = self._parse_single_image(
            origin_image=image,
            prompt_mode=prompt_mode,
            save_dir=save_dir,
            save_name=filename,
            source="pdf",
            page_idx=i,
        )
        result['file_path'] = input_path
        results.append(result)
        
        elapsed = time.time() - page_start
        remaining = (total_pages - i - 1) * elapsed
        print(f"✅ Page {i+1}/{total_pages} done in {elapsed:.1f}s")
        print(f"⏳ Estimated remaining: {remaining:.0f}s ({remaining/60:.1f} min)\n")
    
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"✅ PDF COMPLETE: {total_pages} pages in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"📊 Average: {total_time/total_pages:.1f}s per page")
    print(f"{'='*60}\n")
    
    return results

parser_module.DotsOCRParser.parse_pdf = parse_pdf_with_progress

with open("demo/demo_gradio.py", 'r') as f:
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

demo_code = demo_code.replace('max_height=600,', '')
demo_code = demo_code.replace('show_copy_button=True,', '')
demo_code = demo_code.replace('show_copy_button=False,', '')
demo_code = demo_code.replace('theme="ocean"', 'theme=gr.themes.Soft()')

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

new_layout = '''            with gr.Column(scale=6, variant="compact", elem_id="main_content_column"):
                with gr.Group(elem_id="preview_section"):
                    gr.Markdown("### 👁️ File Preview")
                    result_image = gr.Image(
                        label="Layout Preview",
                        visible=True,
                        height=500,
                        show_label=False,
                        elem_id="result_image"
                    )
                    
                    with gr.Row():
                        prev_btn = gr.Button("⬅ Previous", size="sm")
                        page_info = gr.HTML(
                            value="<div id='page_info_box'>0 / 0</div>", 
                            elem_id="page_info_html"
                        )
                        next_btn = gr.Button("Next ➡", size="sm")
                    
                    info_display = gr.Markdown(
                        "Waiting for processing results...",
                        elem_id="info_box"
                    )
                
                with gr.Group(elem_id="results_section"):
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
                                elem_id="markdown_raw_output",
                                show_label=False
                            )
                        
                        with gr.TabItem("Current Page JSON"):
                            current_page_json = gr.Textbox(
                                value="🕐 Waiting for parsing result...",
                                label="Current Page JSON",
                                max_lines=100,
                                lines=20,
                                elem_id="json_output",
                                show_label=False
                            )'''

demo_code = demo_code.replace(old_layout, new_layout)

css_addition = '''
    
    #main_content_column {
        display: flex;
        flex-direction: column;
        gap: 20px;
    }
    
    #preview_section, #results_section {
        width: 100%;
        max-width: 100%;
        overflow: hidden;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
    }
    
    #markdown_tabs {
        width: 100% !important;
        max-width: 100% !important;
        overflow: hidden !important;
    }
    
    #markdown_output {
        width: 100% !important;
        max-width: 100% !important;
        overflow: auto !important;
        max-height: 600px;
        padding: 10px;
        box-sizing: border-box;
    }
    
    #markdown_output .prose {
        max-width: 100% !important;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }
    
    #markdown_raw_output, #json_output {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    #results_section * {
        max-width: 100%;
        box-sizing: border-box;
    }
'''

demo_code = demo_code.replace('    footer {', css_addition + '\n    footer {')

demo_code = demo_code.replace(
    'demo.launch(server_name="0.0.0.0", server_port=port, debug=True)',
    'demo.launch(server_name="0.0.0.0", server_port=port, show_api=False)'
)

sys.argv = ['demo_gradio.py', '7860']
print("Starting demo with detailed progress tracking...")
exec(demo_code)
