import os
import time
import psutil
import tempfile
from world_model import WorldModel

def test_world_model_full():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_db:
        db_path = tmp_db.name
        
    try:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        wm = WorldModel(db_path=db_path)
        
        # --- TEST 1 & 4: Graph Mutation Stability (Orphans & Consistency) ---
        print("\n[TEST] Graph Mutation Stability...")
        for i in range(100):
            wm.add_node(f"node_{i}", domain="test")
        
        # Add edges
        wm.add_edge("node_1", "node_2", "is")
        wm.add_edge("node_2", "node_3", "is")
        assert wm.graph.number_of_nodes() == 100
        
        # Delete 20 nodes
        for i in range(80, 100):
            wm.delete_node(f"node_{i}")
        assert wm.graph.number_of_nodes() == 80
        
        # Save, load
        wm.save_to_disk()
        wm.load_from_disk()
        
        assert wm.graph.number_of_nodes() == 80
        # Edge test to make sure saving didn't break things
        assert wm.graph.has_edge("node_1", "node_2")
        print("  -> Passed. Save/Load consistent after mutations.")

        # --- TEST 2: Path Explosion Protection & Multi-hop ---
        print("\n[TEST] Path Explosion Protection (Cutoff)..")
        for i in range(20):
            wm.add_edge(f"hop_{i}", f"hop_{i+1}", "causes")
            
        start_time = time.time()
        path = wm.find_path("hop_0", "hop_15", cutoff=10)
        dur = (time.time() - start_time) * 1000
        assert path is None, "Path explosion failed! Cutoff was ignored."
        
        path_valid = wm.find_path("hop_0", "hop_9", cutoff=10)
        assert path_valid is not None, "Valid path under cutoff failed!"
        print(f"  -> Passed in {dur:.2f}ms. Hard cutoff=10 protected deep loops.")

        # --- TEST 3: Transitive Contradiction Propagation ---
        print("\n[TEST] Transitive Contradiction Propagation...")
        wm.add_edge("Alpha", "Beta", "is")
        wm.add_edge("Beta", "Gamma", "is")
        wm.add_edge("Alpha", "Gamma", "is_not")
        
        contradictions = wm.find_contradictions()
        has_transitive = any(c.get("conflict_type") == "transitive_semantic_mismatch" for c in contradictions)
        assert has_transitive, "Transitive contradiction not detected!"
        print("  -> Passed. Detected A->B->C (is) conflicting with A->C (is_not).")

        # --- TEST 4: Temporal Truth Resolution ---
        print("\n[TEST] Temporal Truth Resolution...")
        wm.add_edge("Pluto", "Planet", "is", timestamp=2001.0)
        wm.add_edge("Pluto", "Planet", "is_not", timestamp=2006.0)
        
        # Truth as of 2003
        truth_2003 = wm.query_truth_at_time("Pluto", "Planet", 2003.0)
        assert "is" in truth_2003 and "is_not" not in truth_2003, "Time resolution failed for 2003!"
        
        # Truth as of 2010
        truth_2010 = wm.query_truth_at_time("Pluto", "Planet", 2010.0)
        assert "is_not" in truth_2010 and "is" not in truth_2010, "Time resolution failed for 2010!"
        print("  -> Passed. Correctly filtered historical truths over time.")

        # --- TEST 5: Base RAM Bounds ---
        mem_after = process.memory_info().rss / (1024 * 1024)
        print(f"\n[TEST] Memory Limit Check...")
        print(f"  -> Total RSS: {mem_after:.2f} MB (Limit 500 MB)")
        assert mem_after < 500, "Memory limit exceeded!"
        print("  -> Passed.")
        
        wm.close()

    finally:
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except Exception:
                pass
                
    print("\nALL INVARIANTS VALIDATED SUCCESSFULLY.")

if __name__ == "__main__":
    test_world_model_full()
