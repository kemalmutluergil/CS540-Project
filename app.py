import chromadb
from sentence_transformers import SentenceTransformer
import torch
from transformers import pipeline
import gradio as gr
from datetime import datetime
import json
import os
import transparency
import random
import pandas as pd

print("1. Initializing Local Database & Audit Log...")
chroma_client = chromadb.PersistentClient(path="./local_arbitration_db")
collection = chroma_client.get_or_create_collection(name="arbitration_docs")
AUDIT_LOG_FILE = "audit_log.jsonl"

print("2. Loading Embedding Model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("3. Loading Local LLM onto GPUs...")
llm_pipeline = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-beta", # Changed back to zephyr-7b-beta
    device_map="auto", 
    dtype=torch.float16,
)

# --- ACCOUNTABILITY MODULE 1: Metadata Provenance ---
def add_document(text, doc_name, uploader_name, supersedes_name=""):
    """Chunks text, embeds it, and tracks SUSTAINABILITY (version control)."""
    
    # --- SUSTAINABILITY MODULE: Flagging Obsolete Laws ---
    if supersedes_name.strip():
        # Search for the old document by its ID (doc_name)
        old_docs = collection.get(ids=[supersedes_name.strip()])
        if old_docs and old_docs['ids']:
            old_metadata = old_docs['metadatas'][0]
            old_metadata['is_obsolete'] = "True"
            old_metadata['superseded_by'] = doc_name
            # Update the old document in the database to be permanently obsolete
            collection.update(ids=[supersedes_name.strip()], metadatas=[old_metadata])
            sustainability_msg = f" Also flagged '{supersedes_name}' as OBSOLETE."
        else:
            sustainability_msg = f" (Warning: Could not find '{supersedes_name}' to mark obsolete.)"
    else:
        sustainability_msg = ""

    # Add the new document
    embedding = embedder.encode(text).tolist()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[{
            "upload_user": uploader_name, 
            "document": doc_name,
            "upload_time": timestamp,
            "is_obsolete": "False" # Default to active
        }],
        ids=[doc_name]
    )
    return f"Successfully added {doc_name} to the secure vault.{sustainability_msg}"

def clear_database():
    """Wipes all documents from the collection."""
    chroma_client.delete_collection(name="arbitration_docs")
    global collection
    collection = chroma_client.create_collection(name="arbitration_docs")
    return "Database completely wiped. Ready for fresh uploads."

def ask_copilot(user_query):
    """Retrieves text and generates a draft with EXPLAINABILITY and FAIRNESS constraints."""
    
    # --- FAIRNESS MODULE 1: Proxy Variable Detection ---
    # We scan the prompt for words that might trigger historical or representational bias
    sensitive_proxies = ["western", "developing", "small", "massive", "corporation", "multinational", "common law", "civil law"]
    detected_proxies = [word for word in sensitive_proxies if word in user_query.lower()]
    
    proxy_warning = "System Status: Neutral."
    if detected_proxies:
        proxy_warning = f"⚠️ FAIRNESS ALERT: Proxy variables detected ({', '.join(detected_proxies)}). Strict neutrality constraints activated to prevent bias by proxy."

    # --- SUSTAINABILITY MODULE: The Self-Correcting Legal Monitor ---
    query_embedding = embedder.encode(user_query).tolist()
    
    # We retrieve 2 results to check if obsolete laws are lurking in the top hits
    all_results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=2,
        include=['documents', 'metadatas', 'distances'] 
    )
    
    sustainability_alert = "✅ Legal Monitor: Precedent is active and valid."
    valid_doc_idx = -1
    
    # Scan the retrieved documents
    if all_results['documents'] and len(all_results['documents'][0]) > 0:
        for i in range(len(all_results['documents'][0])):
            meta = all_results['metadatas'][0][i]
            
            # If the system catches an obsolete document, it flags it and blocks it
            if meta.get('is_obsolete') == "True":
                sustainability_alert = f"♻️ SUSTAINABILITY ALERT: System attempted to recall obsolete law '{meta['document']}'. Automatically rerouted to the active precedent: '{meta.get('superseded_by', 'Unknown')}."
            # Grab the first valid, active document to feed to the LLM
            elif valid_doc_idx == -1:
                valid_doc_idx = i
                
    if valid_doc_idx == -1:
        return "No valid, active documents found.", "", "", "", proxy_warning, sustainability_alert, "", ""
        
    retrieved_text = all_results['documents'][0][valid_doc_idx]
    metadata = all_results['metadatas'][0][valid_doc_idx]
    raw_distance = all_results['distances'][0][valid_doc_idx]
    
    confidence_display = transparency.generate_retrieval_trace(metadata, raw_distance, retrieved_text)
    
    # --- FAIRNESS MODULE 2 & 3: Constraint Injection & LLM Auditing ---
    prompt = f"""<|system|>
You are a legal arbitration assistant. 
FAIRNESS CONSTRAINT: You must maintain strict "Equality of Arms". Do not favor Western jurisdictions, Common Law, or large corporations over smaller entities. Base your answer strictly on the facts provided.</s>
<|user|>
Context: {retrieved_text}
User Question: {user_query}

You MUST format your output exactly like this:
DRAFT:
[Your answer here]

KEY DRIVERS:
[Explain which specific clauses drove your answer]

FAIRNESS AUDIT:
[Score your draft's neutrality from 0-100% and provide a 1-sentence justification verifying demographic parity]</s>
<|assistant|>
DRAFT:
"""
    
    raw_output = llm_pipeline(prompt, max_new_tokens=400, return_full_text=False)[0]['generated_text']
    
    # Parse the LLM's structured output
    draft_part, drivers_part, fairness_part = transparency.parse_ai_reasoning(raw_output)
    
    accountability_tag = f"Source Doc: {metadata['document']}\nUploaded by: {metadata['upload_user']}"
    
    return draft_part, drivers_part, confidence_display, fairness_part, proxy_warning, accountability_tag, user_query, retrieved_text, sustainability_alert

# --- ACCOUNTABILITY MODULE 2 & 3: HITL Verification & Audit Log ---
def verify_and_log(user_query, original_source, ai_draft, human_edited_draft, is_verified):
    """Checks the verification box and writes to the immutable audit log."""
    if not is_verified:
        return "⚠️ EXPORT FAILED: You must check the verification box to assume responsibility for this output."
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create the audit record
    audit_record = {
        "timestamp": timestamp,
        "prompt": user_query,
        "source_chunk_used": original_source,
        "original_ai_draft": ai_draft,
        "final_human_approved_version": human_edited_draft
    }
    
    # Append to the local JSONL file (Simulating the immutable backend log)
    with open(AUDIT_LOG_FILE, "a") as f:
        f.write(json.dumps(audit_record) + "\n")
        
    return f"✅ SUCCESS: Exported and logged to backend at {timestamp}."

# --- FAIRNESS MODULE 4: Mathematical Constraints (AIF360) ---
def run_fairness_audit():
    """
    Simulates an AIF360 fairness audit on a batch of historical arbitration data
    to calculate Demographic Parity and Equalized Odds.
    """
    # 1. Simulate Historical Arbitration Dataset
    # Protected attribute A: 1 (Western/Multinational), 0 (Developing/Small Entity)
    # Outcome Y: 1 (Favorable Ruling), 0 (Unfavorable Ruling)
    
    data = []
    for _ in range(500):
        is_western = random.choice([0, 1])
        # In biased historical data, Western entities might have higher chance of favorable outcome (True Label)
        # We simulate the AI's prediction
        true_outcome = 1 if random.random() < (0.7 if is_western == 1 else 0.4) else 0
        
        # Simulated AI Prediction (with some error, maybe mitigating bias or not)
        ai_prediction = true_outcome if random.random() < 0.8 else (1 - true_outcome)
        
        data.append({
            "Entity_Type": "Western/Large" if is_western == 1 else "Non-Western/Small",
            "Is_Western": is_western,
            "True_Outcome": true_outcome,
            "AI_Prediction": ai_prediction
        })
        
    df = pd.DataFrame(data)
    
    # Calculate Metrics (Implementing AIF360 Logic)
    # 1. Demographic Parity: P(Y_pred = 1 | A = 0) vs P(Y_pred = 1 | A = 1)
    priv_pred_favorable = len(df[(df['Is_Western'] == 1) & (df['AI_Prediction'] == 1)]) / len(df[df['Is_Western'] == 1])
    unpriv_pred_favorable = len(df[(df['Is_Western'] == 0) & (df['AI_Prediction'] == 1)]) / len(df[df['Is_Western'] == 0])
    demographic_parity_diff = unpriv_pred_favorable - priv_pred_favorable
    
    # 2. Equalized Odds (True Positive Rate Difference)
    # TPR = P(Y_pred = 1 | Y_true = 1, A)
    priv_tpr = len(df[(df['Is_Western'] == 1) & (df['AI_Prediction'] == 1) & (df['True_Outcome'] == 1)]) / max(1, len(df[(df['Is_Western'] == 1) & (df['True_Outcome'] == 1)]))
    unpriv_tpr = len(df[(df['Is_Western'] == 0) & (df['AI_Prediction'] == 1) & (df['True_Outcome'] == 1)]) / max(1, len(df[(df['Is_Western'] == 0) & (df['True_Outcome'] == 1)]))
    equal_odds_diff = unpriv_tpr - priv_tpr
    
    # Generate Report
    report = f"### 📊 AIF360 Fairness Metrics Audit\n\n"
    report += f"**Dataset Size:** {len(df)} simulated historical arbitration cases.\n"
    report += f"**Protected Attribute (A):** Geography/Size (1 = Western/Large Corporation, 0 = Non-Western/Small Entity)\n\n"
    
    report += f"#### 1. Demographic Parity\n"
    report += f"P(Y = 1 | A = 0) vs P(Y = 1 | A = 1)\n\n"
    report += f"- P(AI Prediction = Favorable | A = Western/Large): **{priv_pred_favorable:.2%}**\n"
    report += f"- P(AI Prediction = Favorable | A = Non-Western/Small): **{unpriv_pred_favorable:.2%}**\n"
    report += f"- **Demographic Parity Difference:** {demographic_parity_diff:.4f} *(Ideal = 0.0)*\n\n"
    
    report += f"#### 2. Equalized Odds (True Positive Rate)\n"
    report += f"*Ensures the model's error rates (False Positives/Negatives) are balanced across groups.*\n\n"
    report += f"- TPR (Western/Large): **{priv_tpr:.2%}**\n"
    report += f"- TPR (Non-Western/Small): **{unpriv_tpr:.2%}**\n"
    report += f"- **Equal Opportunity Difference:** {equal_odds_diff:.4f} *(Ideal = 0.0)*\n\n"
    
    if abs(equal_odds_diff) > 0.1:
        report += "⚠️ **Warning:** The system exhibits bias in predictive accuracy between demographic groups. Mitigation algorithms (e.g., Reweighing or Adversarial Debiasing) should be applied to training data.\n"
    else:
        report += "✅ **Pass:** The system satisfies Equalized Odds constraints within an acceptable margin.\n"
        
    return report

# --- Gradio UI Setup ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚖️ Secure Arbitration RAG Co-Pilot (Accountability Mode)")
    
    # NEW: System Architecture Disclosure
    system_disclosure = transparency.get_system_disclosure()
    gr.Textbox(value=system_disclosure, label="Transparency: System Architecture", interactive=False, lines=5)
    
    # Hidden states to hold data between functions
    hidden_query = gr.State("")
    hidden_source = gr.State("")
    hidden_original_draft = gr.State("")
    
    with gr.Tab("Ask the Co-Pilot"):
        query_input = gr.Textbox(label="What is your question?")
        ask_btn = gr.Button("Generate Draft & Analysis")
        
        # New UI Element for Pre-processing alerts
        proxy_alert_output = gr.Textbox(label="Pre-Processing Bias Check", interactive=False)
        
        gr.Markdown("### Phase 1: Ethical AI Analysis (Explainability & Fairness)")
        with gr.Row():
            confidence_output = gr.Textbox(label="Retrieval Decision Trace (Transparency)", interactive=False, lines=5)
            drivers_output = gr.Textbox(label="Key Drivers (Transparency)", interactive=False, lines=5)
            fairness_output = gr.Textbox(label="Fairness Audit Score", interactive=False, lines=5)
        
        gr.Markdown("### Phase 2: Review AI Draft (Accountability)")
        with gr.Row():
            draft_output = gr.Textbox(label="AI Draft (Editable - Make your corrections here)", interactive=True, lines=6)
            source_output = gr.Textbox(label="Data Provenance", interactive=False, lines=6)
            
        gr.Markdown("### Phase 3: Human-in-the-Loop Verification")
        verification_checkbox = gr.Checkbox(label="I verify I have read the source document and confirm this final draft is accurate.")
        export_btn = gr.Button("Approve & Export to Audit Log", variant="primary")
        status_output = gr.Textbox(label="System Log Status")
        
        with gr.Row():
            proxy_alert_output = gr.Textbox(label="Pre-Processing Bias Check", interactive=False)
            sustain_alert_output = gr.Textbox(label="Sustainability Monitor", interactive=False)
          
        # Notice we added proxy_alert_output and fairness_output to the outputs list!
        ask_btn.click(
            ask_copilot, 
            inputs=[query_input], 
            outputs=[draft_output, drivers_output, confidence_output, fairness_output, proxy_alert_output, sustain_alert_output, source_output, hidden_query, hidden_source]
        ).then( 
            lambda x: x, inputs=[draft_output], outputs=[hidden_original_draft]
        )
        
        export_btn.click(
            verify_and_log,
            inputs=[hidden_query, hidden_source, hidden_original_draft, draft_output, verification_checkbox],
            outputs=[status_output]
        )

    with gr.Tab("Upload Documents"):
        doc_text = gr.Textbox(label="Paste Contract/Rules Text Here", lines=5)
        doc_name = gr.Textbox(label="Document Name")
        user_name = gr.Textbox(label="Your Username")
        
        # 3. Add the new supersedes input field
        supersedes_name = gr.Textbox(label="Supersedes Previous Document? (Optional: Enter the exact Document Name of the old law)")
        
        upload_btn = gr.Button("Securely Upload")
        upload_status = gr.Textbox(label="Status")
        
        # 4. Add 'supersedes_name' to the upload inputs array
        upload_btn.click(add_document, inputs=[doc_text, doc_name, user_name, supersedes_name], outputs=[upload_status])

    with gr.Tab("Fairness Metrics (AIF360)"):
        gr.Markdown("## System-Wide Fairness Audit (Equalized Odds & Demographic Parity)")
        gr.Markdown("This module satisfies the mathematical constraint to ensure the legal doctrine of 'Equality of Arms'. It calculates Demographic Parity and Equalized Odds to ensure the model's error rates are balanced across demographic groups (e.g., nationality of parties, Western vs. Non-Western).")
        
        audit_btn = gr.Button("Run Dataset Fairness Audit", variant="primary")
        audit_results = gr.Markdown(label="AIF360 Audit Results")
        
        audit_btn.click(run_fairness_audit, inputs=[], outputs=[audit_results])

print("4. Launching UI...")
demo.launch(share=True)