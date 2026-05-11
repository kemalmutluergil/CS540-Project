import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import gradio as gr

print("1. Initializing Local Database...")
# Saves the database to a folder in your current directory (no sudo needed)
chroma_client = chromadb.PersistentClient(path="./local_arbitration_db")
collection = chroma_client.get_or_create_collection(name="arbitration_docs")

print("2. Loading Embedding Model...")
# A lightweight model to convert text into searchable vectors
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("3. Loading Local LLM onto GPUs...")
# Using a highly capable open model. 'device_map="auto"' splits it across your 2 GPUs!
# Note: You can change this to Mistral if you prefer.
llm_pipeline = pipeline(
    "text-generation",
    # model="meta-llama/Meta-Llama-3-8B-Instruct", 
    model="HuggingFaceH4/zephyr-7b-beta",
    device_map="auto", 
    dtype=torch.float16,
)

def add_document(text, doc_name, uploader_name):
    """Chunks text, embeds it, and saves it with metadata (Accountability Prep)"""
    # For bare minimum, we just treat the whole text as one chunk for now
    embedding = embedder.encode(text).tolist()
    
    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"upload_user": uploader_name, "document": doc_name}], # METADATA!
        ids=[doc_name]
    )
    return f"Successfully added {doc_name} to the secure vault."

def ask_copilot(user_query):
    """Retrieves relevant text and generates an answer."""
    # 1. Search the database
    query_embedding = embedder.encode(user_query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1)
    
    if not results['documents'][0]:
        return "No relevant documents found in the vault.", ""
        
    retrieved_text = results['documents'][0][0]
    metadata = results['metadatas'][0][0]
    
    # 2. Build the prompt
    prompt = f"""You are a legal arbitration assistant. Answer the user based ONLY on the context below.
    
    Context: {retrieved_text}
    
    User Question: {user_query}
    Answer:"""
    
    # 3. Generate the answer
    output = llm_pipeline(prompt, max_new_tokens=200, return_full_text=False)[0]['generated_text']
    
    # 4. Format output with Accountability tag
    accountability_tag = f"Source: {metadata['document']} | Uploaded by: {metadata['upload_user']}"
    
    return output.strip(), accountability_tag

def clear_database():
    """Wipes all documents from the collection for a fresh test."""
    # Delete the current collection and immediately recreate an empty one
    chroma_client.delete_collection(name="arbitration_docs")
    global collection # Use the global variable so the rest of the app sees the new empty collection
    collection = chroma_client.create_collection(name="arbitration_docs")
    return "Database completely wiped. Ready for fresh uploads."

# --- Gradio UI Setup ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚖️ Secure Arbitration RAG Co-Pilot")
    
    with gr.Tab("Ask the Co-Pilot"):
        query_input = gr.Textbox(label="What is your question?")
        ask_btn = gr.Button("Ask")
        answer_output = gr.Textbox(label="AI Analysis")
        source_output = gr.Textbox(label="Accountability Metadata (Source)")
        ask_btn.click(ask_copilot, inputs=[query_input], outputs=[answer_output, source_output])
        
    with gr.Tab("Upload Documents"):
        doc_text = gr.Textbox(label="Paste Contract/Rules Text Here", lines=5)
        doc_name = gr.Textbox(label="Document Name (e.g., ICC_Rules_2021)")
        user_name = gr.Textbox(label="Your Username")
        upload_btn = gr.Button("Securely Upload")
        upload_status = gr.Textbox(label="Status")
        upload_btn.click(add_document, inputs=[doc_text, doc_name, user_name], outputs=[upload_status])
        # ... your existing upload UI ...
        upload_btn.click(add_document, inputs=[doc_text, doc_name, user_name], outputs=[upload_status])
        
        # Add these two lines right below it:
        gr.Markdown("---") # Just a visual divider line
        clear_btn = gr.Button("🗑️ Clear Entire Database", variant="stop")
        clear_btn.click(clear_database, inputs=[], outputs=[upload_status])

print("4. Launching UI...")
# 'share=True' creates a temporary public link so you can view it on your local browser
demo.launch(share=True)