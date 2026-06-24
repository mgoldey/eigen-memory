import numpy as np
from sklearn.decomposition import IncrementalPCA
import psycopg2
from psycopg2.extensions import register_adapter, AsIs

def addapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
def addapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)
def addapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)
def addapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))

register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)
register_adapter(np.float32, addapt_numpy_float32)
register_adapter(np.int32, addapt_numpy_int32)
register_adapter(np.ndarray, addapt_numpy_array)

class EigenMemoryKernel:
    def __init__(self, db_conn, openai_client, n_components=5):
        self.conn = db_conn
        self.client = openai_client
        self.ipca = IncrementalPCA(n_components=n_components)
        self.buffer = []  # Accumulates vectors for batch updates
        self.buffer_limit = 10
        self.variance_threshold = 0.15

    def add_vector(self, vector):
        """Ingests a surprise vector. If buffer full, runs diagonalization."""
        self.buffer.append(vector)
        if len(self.buffer) >= self.buffer_limit:
            self.add_batch(self.buffer)
            self.buffer = []

    def add_batch(self, vectors):
        """Ingests a batch of vectors directly into PCA and checks for crystallization."""
        if not vectors: return
        
        X = np.array(vectors)
        # IPCA requires n_samples >= n_components? 
        # Actually it handles it, but better to have enough samples.
        self.ipca.partial_fit(X)
        
        # After update, check if we should crystallize
        self._check_and_crystallize()

    def _check_and_crystallize(self):
        """Checks principal components for crystallization (formerly _diagonalize logic)."""
        components = self.ipca.components_
        explained_variance = self.ipca.explained_variance_ratio_
        
        for i, component in enumerate(components):
            if explained_variance[i] > 0.1: # Significant component
                self._crystallize_axiom(component)

    def _crystallize_axiom(self, eigen_vec):
        """Translates a mathematical eigenvector into a linguistic Rule using Contrastive Analysis."""
        
        # 1. Find FAILURES (High Surprise) aligned with this vector
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT context_input, actual_outcome, prediction 
                FROM episodic_buffer 
                WHERE surprise_score > 0.3
                ORDER BY embedding <=> %s::vector 
                LIMIT 3
            """, (eigen_vec.tolist(),))
            failures = cur.fetchall()
            
            # 2. Find SUCCESSES (Low Surprise) aligned with this vector
            # We want cases that are 'similar' in vector space (context) but had low surprise (correct)
            # This helps find the "Near Misses" or the boundary.
            cur.execute("""
                SELECT context_input, actual_outcome 
                FROM episodic_buffer 
                WHERE surprise_score < 0.1
                ORDER BY embedding <=> %s::vector 
                LIMIT 3
            """, (eigen_vec.tolist(),))
            successes = cur.fetchall()
        
        if not failures: 
            return

        # 3. Synthesize Rule with Contrast
        failure_text = "\n".join([f"- Input: {f[0]} | I predicted: {f[2]} | Actual: {f[1]}" for f in failures])
        success_text = "\n".join([f"- Input: {s[0]} | Actual: {s[1]}" for s in successes])
        
        prompt = f"""
        I am making specific mistakes. Help me find the hidden rule.
        
        Here are examples where I FAILED:
        {failure_text}
        
        Here are similar examples where I SUCCEEDED:
        {success_text}
        
        Task: Analyze the difference between these two groups inside <thought> blocks. 
        Then, formulate the SINGLE specific rule or characteristic that makes the 'Failure' cases different from the 'Success' cases.
        Check for arithmetic properties (primes, divisibility, parity, range).
        
        Your final response should be:
        <thought> [Your step-by-step reasoning] </thought>
        RULE: [Concise description of the rule]
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gemma3:4b", 
                messages=[{"role": "user", "content": prompt}]
            )
            axiom = response.choices[0].message.content
            
            # 4. Store in Semantic Core
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO semantic_core (axiom_content, eigen_vector, strength_score)
                    VALUES (%s, %s, %s)
                """, (axiom, eigen_vec.tolist(), 1.0))
            self.conn.commit()
            print(f"Crystallized new axiom: {axiom[:50]}...")
            
        except Exception as e:
            print(f"Failed to crystallize axiom: {e}")
