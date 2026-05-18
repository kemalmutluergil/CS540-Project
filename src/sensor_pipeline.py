import os
import json
import time
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

class SelfCorrectingSensorPipeline:
    def __init__(self, db_path="./local_arbitration_db", collection_name="arbitration_docs"):
        """
        Initializes the Live Sensor. It hooks into the same persistent ChromaDB
        used by your main application to enforce real-time legal sustainability.
        """
        print("[Sensor Initialization] Booting Digital Legal Monitor...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("[Sensor Status] Active. Scanning for legislative changes...")

    def process_incoming_legal_feed(self, update_feed_json: str):
        """
        Simulates scanning official government databases.
        Processes an incoming digital feed stream, identifying if a piece of legislation 
        is an amendment/repeal or a brand new rule.
        """
        try:
            feed_data = json.loads(update_feed_json)
        except json.JSONDecodeError:
            return "❌ SENSOR ERROR: Received malformed legal feed data."

        doc_name = feed_data.get("document_id")
        text = feed_data.get("full_text")
        uploader_name = feed_data.get("issuing_authority", "Gov_Sensor_API")
        supersedes_name = feed_data.get("repealed_statute_id", "").strip()

        print(f"\n📡 [Sensor Triggered] New event detected: Enacting {doc_name}...")

        sustainability_msg = ""
        
        # --- SUSTAINABILITY MONITOR: Auto-detection of Obsolete Precedent ---
        if supersedes_name:
            print(f"⚠️ [Obsolescence Detected] Incoming law '{doc_name}' explicitly replaces '{supersedes_name}'. Searching database...")
            old_docs = self.collection.get(ids=[supersedes_name])
            
            if old_docs and old_docs['ids']:
                old_metadata = old_docs['metadatas'][0]
                old_metadata['is_obsolete'] = "True"
                old_metadata['superseded_by'] = doc_name
                
                # Apply the self-correcting logic directly to the database layer
                self.collection.update(ids=[supersedes_name], metadatas=[old_metadata])
                sustainability_msg = f" | [SYSTEM AUTO-REMEDY] Flagged '{supersedes_name}' as OBSOLETE in database."
                print(f"♻️ {sustainability_msg}")
            else:
                sustainability_msg = f" | [Warning] Detected replacement rule for '{supersedes_name}', but it isn't in local storage yet."
                print(sustainability_msg)

        # Ingest the new legislative update
        embedding = self.embedder.encode(text).tolist()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "upload_user": uploader_name,
                "document": doc_name,
                "upload_time": timestamp,
                "is_obsolete": "False" # Valid by default
            }],
            ids=[doc_name]
        )
        
        return f"⚡ [Sensor Success] Ingested '{doc_name}' seamlessly.{sustainability_msg}"

    def run_stress_test_simulation(self):
        """
        Executes a localized simulation to test the legal environment for time-validity,
        proving that outdated information gets automatically walled off.
        """
        print("\n" + "="*60 + "\n🚀 BEGINNING ETHICAL SUSTAINABILITY STRESS TEST\n" + "="*60)
        
        # Step 1: Ingest an initial framework
        old_law_feed = {
            "document_id": "Arbitration-Act-2018",
            "full_text": "Section 4: All digital cross-border trade disputes must be resolved within 90 days via standard physical arbitration panels.",
            "issuing_authority": "Federal Parliamentary Register"
        }
        print("\n--- Phase 1: Ingesting Initial Baseline Law ---")
        print(self.process_incoming_legal_feed(json.dumps(old_law_feed)))

        # Simulating a lag in amendment passing
        time.sleep(1)

        # Step 2: The Sensor dynamically captures a sudden legislative shift/repeal
        new_amendment_feed = {
            "document_id": "Digital-Trade-Amendment-2026",
            "full_text": "Section 4 of the 2018 Act is completely repealed. All digital disputes are strictly mandated to execute via asynchronous online platforms, extending deadlines to 120 days.",
            "issuing_authority": "Federal Parliamentary Register",
            "repealed_statute_id": "Arbitration-Act-2018"
        }
        print("\n--- Phase 2: Live Digital Sensor Catches a Dynamic Amendment ---")
        print(self.process_incoming_legal_feed(json.dumps(new_amendment_feed)))
        print("\n" + "="*60 + "\n🏁 STRESS TEST SIMULATION COMPLETE\n" + "="*60)


if __name__ == "__main__":
    # Allows you to test the pipeline isolated from the front-end interface
    sensor = SelfCorrectingSensorPipeline()
    sensor.run_stress_test_simulation()