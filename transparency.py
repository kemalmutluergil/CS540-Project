def get_system_disclosure(llm_model="meta-llama/Meta-Llama-3-8B-Instruct", embedder="all-MiniLM-L6-v2"):
    """
    Returns a transparency disclosure about the AI system's architecture.
    """
    disclosure = (
        f"🔍 SYSTEM ARCHITECTURE DISCLOSURE:\n"
        f"- LLM Engine: {llm_model} (Probabilistic Generation)\n"
        f"- Vector Embeddings: {embedder} (Semantic Search)\n"
        f"- Infrastructure: Distributed across available local GPUs.\n"
        f"Notice: This system uses stochastic generation. Outputs may vary and must undergo human-in-the-loop verification."
    )
    return disclosure

def generate_retrieval_trace(metadata, distance, text_chunk):
    """
    Upgrades the raw distance metric into a human-readable audit trace.
    Provides transparency into exactly why a specific piece of data was retrieved.
    """
    word_count = len(text_chunk.split())
    confidence_pct = max(0, min(100, round((1.0 - (distance / 2.0)) * 100, 1)))
    
    threshold_label = "Highly Relevant" if confidence_pct > 75 else "Moderately Relevant" if confidence_pct > 50 else "Low Relevance"
    
    trace = (
        f"🔎 DECISION TRACE:\n"
        f"- Source Document: '{metadata.get('document', 'Unknown')}'\n"
        f"- Chunk Size Analyzed: ~{word_count} words\n"
        f"- Vector Distance: {distance:.4f} (Similarity: {confidence_pct}%)\n"
        f"- System Classification: {threshold_label}\n"
    )
    return trace

def parse_ai_reasoning(raw_output):
    """
    Extracts the structured reasoning, draft, and fairness audit from the LLM.
    If the LLM fails to provide transparency (Key Drivers), it flags it explicitly.
    """
    try:
        draft_part = raw_output.split("KEY DRIVERS:")[0].replace("DRAFT:", "").strip()
        drivers_part = raw_output.split("KEY DRIVERS:")[1].split("FAIRNESS AUDIT:")[0].strip()
        fairness_part = raw_output.split("FAIRNESS AUDIT:")[1].strip()
        
        if not drivers_part:
            drivers_part = "⚠️ TRANSPARENCY FAILURE: The AI failed to articulate its key drivers."
            
    except IndexError:
        draft_part = raw_output
        drivers_part = "⚠️ TRANSPARENCY FAILURE: System could not extract distinct drivers due to formatting mismatch."
        fairness_part = "⚠️ AUDIT FAILURE: Manual review required."
        
    return draft_part, drivers_part, fairness_part
