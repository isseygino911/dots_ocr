FROM rednotehilab/dots.ocr:vllm-openai-v0.9.1

EXPOSE 7860

# The vLLM image runs an API server, so we need to add Gradio
RUN pip install gradio

# Clone the repo to get the demo files
RUN git clone https://github.com/rednote-hilab/dots.ocr.git /app/dots.ocr

WORKDIR /app/dots.ocr

# Run the Gradio demo
CMD ["python", "demo/demo_gradio.py", "--server_name", "0.0.0.0", "--server_port", "7860"]
