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

def run_phase(phase_name, config, seed=42):
    print(f"\n--- Running Phase: {phase_name} (seed={seed}) ---", flush=True)
    print(f"Config: {config}", flush=True)

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

    dataset = generate_dataset(num_samples=DATASET_SIZE, seed=seed)
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
        
        print(f"Processing Batch {b_start // BATCH_SIZE + 1}/ {DATASET_SIZE // BATCH_SIZE}", flush=True)
        
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
        print(f"Batch Accuracy: {batch_acc:.0%}", flush=True)
        
        # Record Metrics
        cumulative_trials += BATCH_SIZE
        phase_data["batch_indices"].append(cumulative_trials)
        phase_data["accuracies"].append(batch_acc)
        phase_data["memory_counts"].append(get_row_count(conn))
        
    # Capture the eigen-spectrum evolution from the Treatment arm for visualization.
    if config["enable_eigen_memory"]:
        phase_data["eigen_spectrum"] = list(agent.kernel.spectrum_history)

    conn.close()
    return phase_data

ARMS = {
    # Baseline: no retrieval, no eigen memory. The DB may grow (learning still
    # records experiences) but nothing is retrieved, so performance should not improve.
    "Baseline": {"enable_retrieval": False, "enable_eigen_memory": False},
    # Control: plain RAG — retrieve top-k similar past episodes, no axioms.
    "Control_RAG": {"enable_retrieval": True, "enable_eigen_memory": False},
    # Treatment: RAG + Eigen-Memory — crystallized axioms injected alongside episodes.
    "Treatment_Eigen": {"enable_retrieval": True, "enable_eigen_memory": True},
}

# Seeds for the fair multi-seed comparison. A single 100-trial run is noisy; we
# average across these so the win/loss direction is not a single-run artifact.
SEEDS = [42, 7, 123]


def main(seeds=SEEDS):
    # results[arm] = {"seeds": {seed: phase_data}, "mean_accuracies": [...]}
    results = {arm: {"seeds": {}} for arm in ARMS}

    for seed in seeds:
        print(f"\n========== SEED {seed} ==========", flush=True)
        for arm, config in ARMS.items():
            results[arm]["seeds"][str(seed)] = run_phase(arm, config, seed=seed)

    # Aggregate: mean accuracy curve and mean memory curve across seeds per arm.
    for arm in ARMS:
        per_seed = list(results[arm]["seeds"].values())
        n_batches = len(per_seed[0]["accuracies"])
        results[arm]["batch_indices"] = per_seed[0]["batch_indices"]
        results[arm]["mean_accuracies"] = [
            float(np.mean([s["accuracies"][b] for s in per_seed])) for b in range(n_batches)
        ]
        results[arm]["std_accuracies"] = [
            float(np.std([s["accuracies"][b] for s in per_seed])) for b in range(n_batches)
        ]
        results[arm]["mean_memory_counts"] = [
            float(np.mean([s["memory_counts"][b] for s in per_seed])) for b in range(n_batches)
        ]
        # Surface a representative eigen-spectrum (first seed) for the heatmap.
        if "eigen_spectrum" in per_seed[0]:
            results[arm]["eigen_spectrum"] = per_seed[0]["eigen_spectrum"]

    with open("comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nComparison Complete. Data saved to comparison_results.json", flush=True)

    # Honest headline: final mean accuracy per arm.
    print("\n=== Final mean accuracy (last batch, across seeds) ===", flush=True)
    for arm in ARMS:
        print(f"  {arm:18s}: {results[arm]['mean_accuracies'][-1]:.0%}", flush=True)


if __name__ == "__main__":
    main()
