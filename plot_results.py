"""Generate the portfolio visuals from comparison_results.json.

Produces:
  - learning_curve.png : cumulative accuracy per arm (mean +/- std band across seeds)
  - memory_cost.png    : memory rows stored per arm over trials
  - eigen_spectrum.png  : evolution of top PCA eigenvalues in the Treatment arm
And prints a real crystallized axiom pulled from the semantic_core table.
"""
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "Baseline": "#888888",
    "Control_RAG": "#1f77b4",
    "Treatment_Eigen": "#2ca02c",
}


def _load():
    with open("comparison_results.json", "r") as f:
        return json.load(f)


def _cumulative(accs):
    """Running mean of per-batch accuracy -> a smooth 'how well learned so far' curve."""
    out, total = [], 0.0
    for i, a in enumerate(accs, start=1):
        total += a
        out.append(total / i)
    return out


def plot_learning_curve(data):
    plt.figure(figsize=(10, 6))
    for arm, results in data.items():
        x = results["batch_indices"]
        cum = _cumulative(results["mean_accuracies"])
        std = results.get("std_accuracies", [0] * len(cum))
        color = COLORS.get(arm, None)
        plt.plot(x, cum, marker="o", label=arm, color=color)
        lo = [max(0, c - s) for c, s in zip(cum, std)]
        hi = [min(1, c + s) for c, s in zip(cum, std)]
        plt.fill_between(x, lo, hi, alpha=0.15, color=color)

    plt.title("Learning Curve: Cumulative Accuracy (mean of seeds, +/- std)")
    plt.xlabel("Cumulative Trials")
    plt.ylabel("Cumulative Accuracy")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=120)
    print("Saved learning_curve.png")


def plot_memory_cost(data):
    plt.figure(figsize=(10, 6))
    for arm, results in data.items():
        plt.plot(
            results["batch_indices"],
            results["mean_memory_counts"],
            marker="x",
            label=arm,
            color=COLORS.get(arm, None),
        )
    plt.title("Memory Cost: Rows Stored over Trials (mean of seeds)")
    plt.xlabel("Cumulative Trials")
    plt.ylabel("DB Rows (Episodic + Semantic)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("memory_cost.png", dpi=120)
    print("Saved memory_cost.png")


def plot_eigen_spectrum(data):
    """Heatmap of the top-k explained-variance ratio recorded over the run.

    Reads results["Treatment_Eigen"]["eigen_spectrum"] if present (list of
    explained_variance_ratio_ snapshots). Skips gracefully if absent.
    """
    treat = data.get("Treatment_Eigen", {})
    spectrum = treat.get("eigen_spectrum")
    if not spectrum:
        print("No eigen_spectrum recorded; skipping eigen_spectrum.png")
        return
    M = np.array(spectrum).T  # rows = components, cols = snapshots over time
    plt.figure(figsize=(10, 4))
    plt.imshow(M, aspect="auto", cmap="viridis", origin="lower")
    plt.colorbar(label="Explained Variance Ratio")
    plt.title("Eigen-Spectrum Evolution (Treatment arm)")
    plt.xlabel("Crystallization Snapshot")
    plt.ylabel("Principal Component")
    plt.tight_layout()
    plt.savefig("eigen_spectrum.png", dpi=120)
    print("Saved eigen_spectrum.png")


def print_example_axiom():
    """Pull one real crystallized axiom from semantic_core for the README."""
    try:
        import psycopg2
        from src.config import get_db_string

        conn = psycopg2.connect(get_db_string())
        with conn.cursor() as cur:
            cur.execute(
                "SELECT axiom_content FROM semantic_core ORDER BY strength_score DESC LIMIT 1"
            )
            row = cur.fetchone()
        conn.close()
        if row:
            print("\n=== Example crystallized axiom ===")
            print(row[0])
        else:
            print("\nNo axioms found in semantic_core.")
    except Exception as e:
        print(f"\nCould not fetch example axiom: {e}")


def main():
    data = _load()
    plot_learning_curve(data)
    plot_memory_cost(data)
    plot_eigen_spectrum(data)
    print_example_axiom()


if __name__ == "__main__":
    main()
