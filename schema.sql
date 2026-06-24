-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Tier 1 & 2: Episodic Buffer (Raw Experience)
CREATE TABLE IF NOT EXISTS episodic_buffer (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    context_input TEXT,
    prediction TEXT,
    actual_outcome TEXT,
    surprise_score FLOAT,           -- The "Loss" (0.0 to 1.0)
    embedding vector(768),         -- Situation vector
    model_state_hash VARCHAR(64),   -- Version control for agent logic
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tier 3: Semantic Core (The "Eigen-Memories")
CREATE TABLE IF NOT EXISTS semantic_core (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    axiom_content TEXT,             -- The compressed rule (e.g., "User hates YAML")
    eigen_vector vector(768),      -- The principal component direction
    strength_score FLOAT DEFAULT 1.0, -- Importance weight
    access_count INT DEFAULT 0,
    last_accessed_at TIMESTAMP DEFAULT NOW()
);

-- High-performance indices
-- Note: 'vector_cosine_ops' assumes normalized vectors for cosine similarity via L2 distance or inner product optimization in newer pgvector versions.
-- We use HNSW for speed.
CREATE INDEX IF NOT EXISTS episodic_buffer_embedding_idx ON episodic_buffer USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS semantic_core_eigen_vector_idx ON semantic_core USING hnsw (eigen_vector vector_cosine_ops);
