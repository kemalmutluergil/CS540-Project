import os
import chromadb
from sentence_transformers import SentenceTransformer

class LegalRAGEngine:
    def __init__(self, db_path: str = "./data/chroma_db"):
        """
        Initializes the sustainable legal database engine.
        Uses local open-source embeddings and a persistent native vector store.
        """
        print("[Initializing] Setting up local embedding model and native vector store...")
        
        # 1. Initialize the local embedding tracker
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # 2. Setup persistent client
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection("legal_knowledge_base")
        print("[Success] Native Legal Database Engine is ready.")

    def add_or_update_law(self, statute_id: str, text: str, metadata: dict):
        """
        Adds a law to the vault. If it exists, updates its context or status tags.
        """
        # Ensure base flags are initialized
        final_metadata = {
            "statute_id": statute_id,
            "is_active": "True",  # Store strings to ensure absolute compatibility with Chroma querying
            **metadata
        }
        
        # If it exists, clear the old copy out first
        try:
            self.collection.delete(ids=[statute_id])
        except Exception:
            pass

        # Encode text locally
        embedding = self.embedder.encode(text).tolist()

        # Save to database
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[final_metadata],
            ids=[statute_id]
        )
        print(f"[Success] Indexed {statute_id} (Active Status: {final_metadata['is_active']})")

    def query_active_law(self, user_query: str):
        """
        Retrieves elements matching semantic criteria where is_active is explicitly 'True'.
        """
        query_embedding = self.embedder.encode(user_query).tolist()
        
        # Execute query utilizing built-in metadata matching conditions
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            where={"is_active": "True"} # Crucial sustainability wall
        )
        
        # Parse return structures for clean viewing
        parsed_docs = []
        if results and results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                parsed_docs.append({
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i]
                })
        return parsed_docs

# --- Isolation Test Suite ---
if __name__ == "__main__":
    engine = LegalRAGEngine()

    print("\n--- Testing: Adding a New Law ---")
    engine.add_or_update_law(
        statute_id="Tax-Rule-2024",
        text="Section 1: All remote workers are subject to a flat 10% digital operations tax.",
        metadata={"jurisdiction": "State-Level", "category": "Tax"}
    )

    print("\n--- Testing: Querying while Active ---")
    active_hits = engine.query_active_law("What is the tax rate for remote workers?")
    for item in active_hits:
        print(f"-> Found Valid Record: {item['text']} (Metadata: {item['metadata']})")

    print("\n--- Testing: Sensor Flags Law as Obsolete ---")
    engine.add_or_update_law(
        statute_id="Tax-Rule-2024",
        text="Section 1: All remote workers are subject to a flat 10% digital operations tax.",
        metadata={"jurisdiction": "State-Level", "category": "Tax", "is_active": "False"}
    )

    print("\n--- Testing: Querying after Obsolescence ---")
    stale_hits = engine.query_active_law("What is the tax rate for remote workers?")
    if not stale_hits:
        print("✅ SUCCESS: The system completely blocked the obsolete law from being retrieved!")
    else:
        print("❌ FAILURE: Stale rules leaked out.")
