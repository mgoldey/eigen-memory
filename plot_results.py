import json
import matplotlib.pyplot as plt

def plot_results():
    try:
        with open("comparison_results.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("comparison_results.json not found.")
        return

    # Plot 1: Accuracy over Batches
    plt.figure(figsize=(10, 6))
    for phase, results in data.items():
        batches = range(1, len(results["accuracies"]) + 1)
        plt.plot(batches, results["accuracies"], marker='o', label=phase)
    
    plt.title("Agent Accuracy over Time (Batches)")
    plt.xlabel("Batch Number (size=10)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy_plot.png")
    print("Saved accuracy_plot.png")

    # Plot 2: Memory Usage (DB Rows) over Time
    plt.figure(figsize=(10, 6))
    for phase, results in data.items():
        # Memory count is recorded after each batch
        # We want to see how the DB grows.
        counts = results["memory_counts"]
        cum_trials = results["batch_indices"]
        plt.plot(cum_trials, counts, marker='x', label=phase)

    plt.title("Memory Growth (DB Rows) over Trials")
    plt.xlabel("Cumulative Trials")
    plt.ylabel("DB Row Count (Episodic + Semantic)")
    plt.legend()
    plt.grid(True)
    plt.savefig("memory_plot.png")
    print("Saved memory_plot.png")

if __name__ == "__main__":
    plot_results()
