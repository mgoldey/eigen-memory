import hashlib
import numpy as np
from openai import OpenAI
import psycopg2
from .memory_kernel import EigenMemoryKernel

try:
    from src.config import OLLAMA_BASE_URL
except ImportError:  # allow import when run as a standalone module
    OLLAMA_BASE_URL = "http://localhost:11434/v1"

# --- Surprise / memory thresholds (in nats of NLL) ---
# Surprise assigned when the true-label token is absent from the top-k logprobs.
# Floors a "missing" token at a high-but-finite value so it reliably counts as
# surprising without the old bug where this case silently collapsed to a flat 2.0.
MISSING_TOKEN_NLL = 7.0
# An experience is written to the episodic buffer when its predictive surprise
# (or perceptual salience) clears this bar. Tuned to the observed NLL distribution
# of gemma3:4b on this task so that genuine mistakes are recorded.
EPISODIC_WRITE_NLL = 1.0
# A surprise vector feeds the Eigen kernel (and may crystallize an axiom) above
# this bar. Set higher than the write bar so axioms form from the *most* surprising
# events, not every recorded one.
EIGEN_CRYSTALLIZE_NLL = 1.5
# Cap on the prediction call's chain-of-thought tokens. Enough room for a short
# <thought> block plus the final label line, without the ~1.3k-char ramble that
# makes each call slow and tends to make a 4B model drift.
PREDICTION_MAX_TOKENS = 200


def _extract_nll(top_logprobs, true_label):
    """Negative log-likelihood (nats) of the true label among top-k logprobs.

    Returns MISSING_TOKEN_NLL when the true-label token does not appear in the
    top-k, rather than leaving the value undefined (the original bug).
    """
    for tl in top_logprobs:
        if true_label.upper() in tl.token.upper():
            return float(-tl.logprob)
    return MISSING_TOKEN_NLL


# System prompt for the predictive-surprise probe. gemma3:4b is a CHAT model: a
# completion-style prompt ("...the label:") makes it reply with prose ("The...",
# "In...") so the label token is never first and NLL collapses to MISSING_TOKEN_NLL
# for every item. Constraining it to emit exactly one label word puts the label as
# the first token, so its logprob is a real, varied prediction-error signal.
DEFAULT_LABELS = ["RED", "BLUE", "GREEN"]


def _surprise_messages(query, labels=DEFAULT_LABELS):
    """Messages for the predictive-surprise probe (label must be the first token)."""
    label_str = ", ".join(labels)
    return [
        {
            "role": "system",
            "content": (
                "You are playing a classification game. Reply with EXACTLY one "
                f"word, one of: {label_str}. No other text."
            ),
        },
        {"role": "user", "content": f"Input: {query}\nLabel:"},
    ]


class AgenticMemoryLoop:
    def __init__(self, db_string, openai_client=None, model="gemma3:4b", thought_model="gemma3:4b", enable_retrieval=True, enable_eigen_memory=True, labels=None):
        self.conn = psycopg2.connect(db_string)
        # Valid label set for this task (RED/BLUE/GREEN by default; e.g.
        # HUM/LOC/NUM for TREC). Used in both the prediction and surprise prompts.
        self.labels = labels or DEFAULT_LABELS
        # If client not provided, default to local Ollama
        if openai_client is None:
            self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        else:
            self.client = openai_client
            
        self.model = model
        self.thought_model = thought_model
        self.enable_retrieval = enable_retrieval
        self.enable_eigen_memory = enable_eigen_memory
        self.kernel = EigenMemoryKernel(self.conn, self.client, model=self.thought_model)
        self.state_hash = self._get_hash()

    def _get_hash(self):
        return hashlib.sha256(f"{self.model}".encode()).hexdigest()

    def _embed(self, text):
        # Use a dedicated embedding model via Ollama
        try:
             # We use 'embeddinggemma:latest' as requested
             res = self.client.embeddings.create(input=str(text), model="embeddinggemma:latest")
             return np.array(res.data[0].embedding)
        except Exception as e:
            # Fallback: a random vector keeps the run alive, but it turns retrieval
            # into noise for this item. Count failures so a degraded run is visible
            # rather than silently corrupting results.
            self.embed_failures = getattr(self, "embed_failures", 0) + 1
            print(f"Warning: Embedding failed ({e}), using random vector. "
                  f"(total embed failures: {self.embed_failures})")
            return np.random.rand(768)


    def run_batch(self, inputs):
        """Runs a batch of inputs with Perceptual Surprise (Entropy) and CoT."""
        predictions = []
        embeddings = []
        salience_flags = []
        perceptual_surprises = []
        
        for x in inputs:
            q_vec = self._embed(x)
            embeddings.append(q_vec)
            context = self._retrieve_context(q_vec)
            
            # 2. Predict with CoT and Logprobs
            label_str = ", ".join(self.labels)
            prompt = f"""You are playing a pattern recognition game.
            Valid outputs are only: {label_str}.

            {context}

            Task: Analyze the input and rules inside <thought> blocks.
            Your final line must be exactly one of: {label_str}.

            Input: {x}
            Output:"""
            
            try:
                pred_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    logprobs=True,
                    top_logprobs=5,  # To see distribution across labels
                    # Cap the chain-of-thought. Uncapped, gemma3:4b emits ~1.3k chars
                    # of reasoning per call (slow, and long CoT tends to drift on a
                    # 4B model). A short budget keeps the final RED/BLUE/GREEN line.
                    max_tokens=PREDICTION_MAX_TOKENS,
                )
                raw = pred_resp.choices[0].message.content
                predictions.append(raw)
                
                # Calculate Perceptual Surprise (Entropy of the first token)
                if pred_resp.choices[0].logprobs and pred_resp.choices[0].logprobs.content:
                    logprobs_info = pred_resp.choices[0].logprobs.content
                    top = logprobs_info[0].top_logprobs
                    probs = [np.exp(tl.logprob) for tl in top]
                    entropy = -np.sum([p * np.log(p + 1e-10) for p in probs])
                    perceptual_surprises.append(entropy)
                    salience_flags.append(entropy > 0.8)
                else:
                    print(f"Warning: No logprobs in prediction response. logprobs={pred_resp.choices[0].logprobs}")
                    perceptual_surprises.append(0.0)
                    salience_flags.append(False)

            except Exception as e:
                print(f"Prediction Error: {e}")
                predictions.append("ERROR")
                perceptual_surprises.append(0.0)
                salience_flags.append(False)
                
        return predictions, embeddings, salience_flags, perceptual_surprises

    def learn_batch(self, inputs, raw_predictions, true_labels, embeddings, salience_flags):
        """Calculates Predictive Surprise (NLL) and updates kernel."""
        predictive_surprises = []
        surprising_vectors = []
        
        for i, (query, pred, outcome, q_vec, is_salient) in enumerate(zip(inputs, raw_predictions, true_labels, embeddings, salience_flags)):
            # Predictive Surprise: How likely was the TRUE label? We probe the model
            # for a one-token label answer and read the logprob of the true label.
            # The constrained prompt forces the label to be the first token (see
            # _surprise_messages); a completion-style prompt fails on chat models.
            try:
                res = self.client.chat.completions.create(
                    model=self.model,
                    messages=_surprise_messages(query, self.labels),
                    logprobs=True,
                    top_logprobs=10,
                    max_tokens=1
                )
                
                if res.choices[0].logprobs and res.choices[0].logprobs.content:
                    lp_content = res.choices[0].logprobs.content[0]
                    # NLL of the true label among top-k. If the true-label token is
                    # absent from top-k, _extract_nll returns a high, finite value
                    # (the original code left target_lp undefined here and silently
                    # collapsed every miss to a flat 2.0 via the except path).
                    s_pred = _extract_nll(lp_content.top_logprobs, outcome)
                else:
                    # Fallback to Embedding Surprise if logprobs missing
                    # We need to extract the color from the CoT prediction
                    lines = [l.strip() for l in pred.strip().split("\n") if l.strip()]
                    pred_color = lines[-1].upper().replace(".", "").replace("'","").replace('"',"") if lines else "ERROR"
                    
                    v_pred = self._embed(pred_color)
                    v_out = self._embed(outcome)
                    s_pred = 1 - np.dot(v_pred, v_out)
                    # Scale s_pred to bit-like range (0.5 cos -> roughly 1.0-2.0 bits)
                    # Just keep it raw for now but ensure it triggers learning
                    s_pred = s_pred * 5.0 
                
                predictive_surprises.append(s_pred)
                
            except Exception as e:
                print(f"Surprise Eval Error: {e}")
                s_pred = 2.0 # Manual high surprise fallback
                predictive_surprises.append(s_pred)

            # Store in Episodic Buffer if either Salient (Perceptual) or Surprising (Predictive)
            if is_salient or s_pred > EPISODIC_WRITE_NLL:
                try:
                    with self.conn.cursor() as cur:
                        cur.execute("INSERT INTO episodic_buffer (context_input, prediction, actual_outcome, surprise_score, embedding) VALUES (%s, %s, %s, %s, %s)",
                        (query, pred, outcome, float(s_pred), q_vec.tolist()))
                    self.conn.commit()
                except Exception as e:
                    print(f"Logging Error: {e}")
                    self.conn.rollback()

            if s_pred > EIGEN_CRYSTALLIZE_NLL: # High predictive error drives Eigen-Memory
                surprising_vectors.append(q_vec)

        # Batch Update Kernel
        if self.enable_eigen_memory and surprising_vectors:
             self.kernel.add_batch(surprising_vectors)
        
        print(f"Batch Predictive Surprise (Avg NLL): {np.mean(predictive_surprises):.2f}")

    def _retrieve_context(self, vec):
        if not self.enable_retrieval:
            return ""

        ctx_str = ""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT context_input, actual_outcome, surprise_score
                FROM episodic_buffer
                ORDER BY embedding <=> %s::vector
                LIMIT 3
            """, (vec.tolist(),))
            episodes = cur.fetchall()
            
            if episodes:
                ctx_str += "Similar Past Events:\n"
                for ep in episodes:
                    ctx_str += f"- Input: {ep[0]} -> Result: {ep[1]} (Surprise: {ep[2]:.2f})\n"
            
            # Only the Eigen arm injects crystallized axioms. Gating here keeps the
            # Control_RAG arm a pure episodic-retrieval baseline even if axioms from
            # a prior phase linger in the table.
            if self.enable_eigen_memory:
                cur.execute("""
                    SELECT axiom_content, strength_score
                    FROM semantic_core
                    ORDER BY eigen_vector <=> %s::vector
                    LIMIT 2
                """, (vec.tolist(),))
                axioms = cur.fetchall()

                if axioms:
                    print(f"[AXIOM→] injected {len(axioms)} axiom(s) into context")
                    ctx_str += "\nRelevant Rules/Memories:\n"
                    for ax in axioms:
                        ctx_str += f"- RULE: {ax[0]}\n"

        return ctx_str if ctx_str else "No relevant memories found."

    def run(self, user_query):
        # Implementation of single-run legacy-compat if needed
        return self.run_batch([user_query])[0][0]
