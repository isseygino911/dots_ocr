# Use the official dots.ocr Docker image
FROM rednotehilab/dots.ocr:latest

# Expose Gradio port
EXPOSE 7860

# Run the demo
CMD ["python", "demo/demo_gradio.py", "--server_name", "0.0.0.0", "--server_port", "7860"]
