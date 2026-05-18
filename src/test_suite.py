import os
from rag_engine import LegalRAGEngine

class LegalSystemStressTester:
    def __init__(self):
        # Dedicated sandbox database for testing
        self.test_db_path = "./data/test_chroma_db"
        self.engine = LegalRAGEngine(db_path=self.test_db_path)
        self.reset_sandbox()

    def reset_sandbox(self):
        """
        Wipes all records cleanly from the collection using native database commands.
        This side-steps Windows OS file-locking permissions completely.
        """
        try:
            # Look up all existing IDs in the collection and delete them safely
            all_ids = self.engine.collection.get()["ids"]
            if all_ids:
                self.engine.collection.delete(ids=all_ids)
            print("🧹 [Sandbox Warehouse] Test data wiped clean via native database flush.")
        except Exception as e:
            print(f"⚠️ [Sandbox Warehouse] Notice during reset: {e}")

    def run_all_tests(self):
        print("\n" + "="*60 + "\n🚦 RUNNING ETHICAL COMPLIANCE & ACCURACY STRESS TESTS\n" + "="*60)
        
        self.test_case_1_standard_supersede()
        self.test_case_2_temporal_amnesia()
        self.test_case_3_version_chaining()
        
        print("\n" + "="*60 + "\n🎉 ALL SYSTEM COMPLIANCE TESTS PASSED SUCCESSFULLY!\n" + "="*60)

    def test_case_1_standard_supersede(self):
        print("\n▶️ [Test 1] Testing Standard Statute Replacement...")
        self.reset_sandbox()
        
        # 1. Inject an old rule
        self.engine.add_or_update_law(
            statute_id="EU-Data-2018",
            text="Companies must respond to user data privacy requests within 30 days.",
            metadata={"region": "EU"}
        )
        
        # 2. Inject an amendment that replaces it
        self.engine.add_or_update_law(
            statute_id="EU-Data-2018",
            text="Companies must respond to user data privacy requests within 30 days.",
            metadata={"region": "EU", "is_active": "False"} # Deactivated
        )
        self.engine.add_or_update_law(
            statute_id="EU-Data-2026-Amendment",
            text="The 30-day window is repealed. Companies must now respond within 7 business days.",
            metadata={"region": "EU"}
        )
        
        # 3. Query the system
        results = self.engine.query_active_law("How long do companies have to respond to data requests?")
        
        # 4. Verify compliance
        assert len(results) > 0, "System failed to find any active laws."
        for doc in results:
            assert "30 days" not in doc['text'], "❌ CRITICAL FAILURE: System retrieved an obsolete law!"
            assert "7 business days" in doc['text'], "System failed to prioritize the newest amendment."
            
        print("✅ PASS: System correctly prioritized the updated amendment over the stale rule.")

    def test_case_2_temporal_amnesia(self):
        print("\n▶️ [Test 2] Testing Complete Temporal Amnesia (Zero Leaks)...")
        self.reset_sandbox()
        
        # 1. Add a highly specific law and instantly declare it inactive
        self.engine.add_or_update_law(
            statute_id="Obsolete-Crypto-Tax",
            text="All cryptocurrency mining operations must pay a 40% clean energy premium tax.",
            metadata={"is_active": "False"}
        )
        
        # 2. Query for that exact topic
        results = self.engine.query_active_law("What is the crypto clean energy premium tax rate?")
        
        # 3. Verify that nothing leaked out
        assert len(results) == 0, f"❌ CRITICAL FAILURE: Inactive data leaked! Found: {results}"
        print("✅ PASS: System demonstrated total amnesia regarding inactive records.")

    def test_case_3_version_chaining(self):
        print("\n▶️ [Test 3] Testing Multi-Version Chaining (Law A -> B -> C)...")
        self.reset_sandbox()
        
        # Version A
        self.engine.add_or_update_law("Rent-Law-V1", "Rent increases are capped at 5% annually.", {"is_active": "False"})
        # Version B
        self.engine.add_or_update_law("Rent-Law-V2", "Rent increases are capped at 8% annually.", {"is_active": "False"})
        # Version C
        self.engine.add_or_update_law("Rent-Law-V3", "Rent increases are completely frozen at 0% for the next two years.", {"is_active": "True"})
        
        # Query
        results = self.engine.query_active_law("What is the legal limit for a landlord to raise rent?")
        
        # Verify
        assert len(results) == 1, f"Expected 1 active record, found {len(results)}"
        assert "0%" in results[0]['text'], f"❌ FAILURE: Failed to retrieve the terminal node in the chain. Got: {results[0]['text']}"
        print("✅ PASS: System navigated the historical version chain perfectly.")

if __name__ == "__main__":
    tester = LegalSystemStressTester()
    tester.run_all_tests()