# Experimental Design Document (EDD)
**Title**: Efficacy of Iterative Diagonalization in Agentic Memory Consolidation
**Date**: January 2026

> This is the **pre-registered hypothesis** (the plan, written before results). For what
> actually happened — and a critique of whether this design could even test H1 — see
> [FINDINGS.md](FINDINGS.md). Kept here unaltered so the original hypothesis is on record.

## 1. Abstract
This experiment validates whether an "Eigen-Memory" system (Treatment) reduces "Surprise" (Prediction Error) faster than a standard RAG system (Control) in a sequential decision-making environment with hidden rules.

## 2. Hypothesis
$H_1$: The Eigen-Memory system will exhibit a statistically significant reduction in the Area Under the Surprise Curve (AUSC) compared to standard RAG, indicating faster induction of hidden rules.

## 3. Methodology: The "Hidden Rules" Game
We will use a synthetic dataset where the agent must classify numbers, but the rules are hidden and change based on context.

### Task
"Classify the input $X$."

### Hidden Rule
"If $X$ is prime, output 'RED'. If $X$ is divisible by 5, output 'BLUE'. Else 'GREEN'."

### Process
1. The agent guesses.
2. We give feedback (Correct/Incorrect).
3. The agent remembers.

### Groups
- **Group A (Control - RAG)**: Stores every interaction. Retrieves top-3 similar past interactions.
- **Group B (Treatment - Eigen)**: Uses the `AgenticMemoryLoop`. Compresses failures into Axioms using Iterative PCA. Retrieves top-1 Axiom + top-1 Episode.

## 4. Metrics
- **AUSC (Area Under Surprise Curve)**: Sum of surprise scores over 100 trials. Lower is better.
- **Tokens to Convergence**: How many tokens does the agent consume before it answers 10 consecutive inputs correctly?
- **Context Saturation**: The number of characters in the prompt context at Trial #100.

## 5. Visualizations to Generate
- **The Learning Curve**: X-Axis: Trial Number (1-100). Y-Axis: Surprise Score. (Group A should be jagged/flat; Group B should decay exponentially).
- **The Eigen-Spectrum**: A heatmap showing the evolution of the top 3 Eigenvectors in Group B.

## 6. Implementation References
- **Agent Loop**: [src/eigen_memory_agent/agent.py](src/eigen_memory_agent/agent.py)
- **Memory Kernel**: [src/eigen_memory_agent/memory_kernel.py](src/eigen_memory_agent/memory_kernel.py)
- **Simulation Script**: `simulate.py` (To be created)
