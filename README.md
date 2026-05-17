## Prerequisites
* Python 3.8+
* A local environment or server with GPU access (the script is configured to auto-distribute the LLM across available GPUs).

## Installation

It is highly recommended to run this within a Python virtual environment. Install the required dependencies using the following command:

```bash
pip install torch transformers accelerate sentence-transformers chromadb gradio huggingface_hub pandas
```

To run: 
```bash
python3 app.py
```

You need an HF access token to run the default Llama model, you can use another model like zephyr that does not require a token by changig the following line:
```
print("3. Loading Local LLM onto GPUs...")
llm_pipeline = pipeline(
    "text-generation",
    model="meta-llama/Meta-Llama-3-8B-Instruct", # CHANGE THIS "HuggingFaceH4/zephyr-7b-beta"
    device_map="auto", 
    dtype=torch.float16,
)
```
