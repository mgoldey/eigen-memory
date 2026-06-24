# Eigen-Memory Agent

Implementing and evaluating a lossy eigen-memory agent with Gemma 3 4b + pgvector.

## Setup

1.  **Dependencies**:
    ```bash
    uv sync
    ```

2.  **Infrastructure**:
    ```bash
    docker-compose up -d
    # Ensure Ollama is running locally
    ollama run gemma3:4b
    ```

3.  **Database**:
    ```bash
    psql -h localhost -U postgres -d memory_agent -f schema.sql
    ```
    (Password: `password`)

## Running

1.  **Baseline Simulation**:
    ```bash
    uv run simulate.py
    ```

## Architecture
- **Episodic Buffer**: Raw experience log in Postgres.
- **Semantic Core**: Eigen-vectors of high-surprise events.
- **Kernel**: Iterative PCA for memory consolidation.
