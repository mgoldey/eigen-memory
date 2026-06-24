import json
import time
import psycopg2
import numpy as np
from src.eigen_memory_agent.agent import AgenticMemoryLoop
from src.dataset import generate_dataset
from src.config import get_db_string

# Configuration
DB_STRING = get_db_string()
DATASET_SIZE = 100 # Apples to Apples: 100 trials each
BATCH_SIZE = 10

def get_row_count(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT (SELECT COUNT(*) FROM episodic_buffer) + (SELECT COUNT(*) FROM semantic_core)")
        return cur.fetchone()[0]

def run_phase(phase_name, config):
    print(f"\n--- Running Phase: {phase_name} ---")
    print(f"Config: {config}")
    
    # 1. Reset Database
    conn = psycopg2.connect(DB_STRING)
    with conn.cursor() as cur:
        cur.execute("TRUNCATE episodic_buffer, semantic_core;")
    conn.commit()
    
    # 2. Initialize Agent with Config
    agent = AgenticMemoryLoop(
        DB_STRING, 
        model="gemma3:4b", 
        thought_model="gemma3:4b",
        enable_retrieval=config["enable_retrieval"],
        enable_eigen_memory=config["enable_eigen_memory"]
    )
    
    dataset = generate_dataset(num_samples=DATASET_SIZE)
    phase_data = {
        "batch_indices": [],
        "accuracies": [],
        "memory_counts": [], # Rows in DB
        "mses": []
    }
    
    total_score = 0
    cumulative_trials = 0
    
    # Batch Loop
    for b_start in range(0, DATASET_SIZE, BATCH_SIZE):
        batch = dataset[b_start : b_start + BATCH_SIZE]
        inputs = [item['input'] for item in batch]
        true_labels = [item['label'] for item in batch]
        
        print(f"Processing Batch {b_start // BATCH_SIZE + 1}/ {DATASET_SIZE // BATCH_SIZE}")
        
        # Run Batch
        raw_predictions, embeddings, salience_flags, p_surprises = agent.run_batch(inputs)
        
        # Learn Batch
        agent.learn_batch(inputs, raw_predictions, true_labels, embeddings, salience_flags)
        
        # Score
        batch_correct = 0
        clean_preds = []
        for raw in raw_predictions:
            # Extract last line for the color
            lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
            p = lines[-1].upper().replace(".", "").replace("'","").replace('"',"") if lines else "ERROR"
            if p not in ["RED", "BLUE", "GREEN"]:
                if "RED" in raw.upper(): p = "RED"
                elif "BLUE" in raw.upper(): p = "BLUE"
                elif "GREEN" in raw.upper(): p = "GREEN"
            clean_preds.append(p)
            
        for p, t in zip(clean_preds, true_labels):
            if p == t: batch_correct += 1
            
        batch_acc = batch_correct / BATCH_SIZE
        print(f"Batch Accuracy: {batch_acc:.0%}")
        
        # Record Metrics
        cumulative_trials += BATCH_SIZE
        phase_data["batch_indices"].append(cumulative_trials)
        phase_data["accuracies"].append(batch_acc)
        phase_data["memory_counts"].append(get_row_count(conn))
        
    conn.close()
    return phase_data

def main():
    results = {}
    
    # 1. Baseline (No Retrieval, No Memory)
    # Actually, Baseline "No Memory" means we don't retrieve. 
    # Do we learn? If we learn but don't retrieve, the DB grows but performance shouldn't change.
    # That's a good baseline.
    results["Baseline"] = run_phase("Baseline", {"enable_retrieval": False, "enable_eigen_memory": False})
    
    # 2. Control RAG (Retrieval ON, Eigen OFF)
    results["Control_RAG"] = run_phase("Control_RAG", {"enable_retrieval": True, "enable_eigen_memory": False})
    
    # 3. Treatment (Retrieval ON, Eigen ON)
    results["Treatment_Eigen"] = run_phase("Treatment_Eigen", {"enable_retrieval": True, "enable_eigen_memory": True})
    
    # Save
    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nComparison Complete. Data saved to comparison_results.json")

if __name__ == "__main__":
    main()
