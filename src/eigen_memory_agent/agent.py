import hashlib
import json

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
# Cap on the prediction call's chain-of-thought tokens. Enough room for a short
# <thought> block plus the final label line, without the ~1.3k-char ramble that
# makes each call slow and tends to make a 4B model drift.
PREDICTION_MAX_TOKENS = 200
# How many axioms to inject into context (Treatment arm only).
AXIOM_INJECT_TOP_K = 2


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


def _surprise_messages(query, labels=DEFAULT_LABELS, context=""):
    """Messages for the predictive-surprise probe (label must be the first token).

    The retrieved memory context is included so the probe measures the prediction
    error of the FULL AGENT, not the bare model (THEORY.md section 6): items the
    memory already handles stop registering as surprising, which is what makes
    consolidation self-limiting.
    """
    label_str = ", ".join(labels)
    user = f"Input: {query}\nLabel:"
    if context:
        user = f"{context}\n\n{user}"
    return [
        {
            "role": "system",
            "content": (
                "You are playing a classification game. Reply with EXACTLY one "
                f"word, one of: {label_str}. No other text."
            ),
        },
        {"role": "user", "content": user},
    ]


def clean_prediction(raw, labels):
    """Extract the predicted label from a raw CoT response.

    Take the last non-empty line; if it isn't a valid label, fall back to the
    EARLIEST occurrence of any valid label in the text (position order, not
    label-list order — the old fallback scanned in label-list order, which
    systematically biased truncated responses toward the first label).
    """
    lines = [l.strip() for l in raw.strip().split("\n") if l.strip()]
    p = (
        lines[-1].upper().replace(".", "").replace("'", "").replace('"', "")
        if lines
        else "ERROR"
    )
    if p not in labels:
        up = raw.upper()
        hits = [(up.find(lab), lab) for lab in labels if lab in up]
        if hits:
            p = min(hits)[1]
    return p


class AgenticMemoryLoop:
    def __init__(self, db_string, openai_client=None, model="gemma3:4b", thought_model="gemma3:4b", enable_retrieval=True, enable_eigen_memory=True, labels=None, static_context=""):
        self.conn = psycopg2.connect(db_string)
        # Valid label set for this task (RED/BLUE/GREEN by default; e.g.
        # HUM/LOC/NUM for TREC). Used in both the prediction and surprise prompts.
        self.labels = labels or DEFAULT_LABELS
        # Fixed context prepended to every prediction (e.g. the true rule for an
        # Oracle_Rule ceiling arm). Empty for normal arms.
        self.static_context = static_context
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
        """Runs a batch of inputs with Perceptual Surprise (Entropy) and CoT.

        Returns per-item lists: raw predictions, embeddings, salience flags,
        perceptual surprises, retrieved contexts (needed by the memory-conditional
        surprise probe), and retrieval residuals (query embedding minus nearest
        retrieved embedding; None when nothing was retrieved).
        """
        predictions = []
        embeddings = []
        salience_flags = []
        perceptual_surprises = []
        contexts = []
        residuals = []

        for x in inputs:
            q_vec = self._embed(x)
            embeddings.append(q_vec)
            context, nn_vec = self._retrieve_context(q_vec)
            if self.static_context:
                context = self.static_context + ("\n\n" + context if context else "")
            contexts.append(context)
            residuals.append(q_vec - nn_vec if nn_vec is not None else None)

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

        return predictions, embeddings, salience_flags, perceptual_surprises, contexts, residuals

    def learn_batch(self, inputs, raw_predictions, true_labels, embeddings, salience_flags,
                    contexts=None, residuals=None):
        """Measures memory-conditional predictive surprise (NLL), stores episodes,
        and feeds retrieval residuals to the consolidation kernel."""
        contexts = contexts if contexts is not None else [""] * len(inputs)
        residuals = residuals if residuals is not None else [None] * len(inputs)
        predictive_surprises = []

        for query, pred, outcome, q_vec, is_salient, context, residual in zip(
            inputs, raw_predictions, true_labels, embeddings, salience_flags, contexts, residuals
        ):
            pred_label = clean_prediction(pred, self.labels)
            was_correct = pred_label == outcome

            # Predictive Surprise: how likely was the TRUE label *given the same
            # memory context the agent predicted with*? Probing without the context
            # (the old behavior) measured the bare model's difficulty, so items the
            # memory already handled kept registering as surprising and the signal
            # could never decline as the agent learned.
            try:
                res = self.client.chat.completions.create(
                    model=self.model,
                    messages=_surprise_messages(query, self.labels, context),
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
                    v_pred = self._embed(pred_label)
                    v_out = self._embed(outcome)
                    s_pred = (1 - np.dot(v_pred, v_out)) * 5.0

                predictive_surprises.append(s_pred)

            except Exception as e:
                print(f"Surprise Eval Error: {e}")
                s_pred = 2.0 # Manual high surprise fallback
                predictive_surprises.append(s_pred)

            # Store in Episodic Buffer if either Salient (Perceptual) or Surprising (Predictive)
            if is_salient or s_pred > EPISODIC_WRITE_NLL:
                try:
                    with self.conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO episodic_buffer (context_input, prediction, actual_outcome, surprise_score, embedding, was_correct) VALUES (%s, %s, %s, %s, %s, %s)",
                            (query, pred, outcome, float(s_pred), q_vec.tolist(), was_correct),
                        )
                    self.conn.commit()
                except Exception as e:
                    print(f"Logging Error: {e}")
                    self.conn.rollback()

            # Feed the kernel EVERY retrieval-bearing trial, keyed on correctness —
            # not gated on surprise. The covariance contrast needs unbiased samples
            # of both failures and successes (THEORY.md section 3).
            if self.enable_eigen_memory and residual is not None:
                self.kernel.observe(
                    embedding=q_vec,
                    residual=residual,
                    was_correct=was_correct,
                    context_input=query,
                    prediction=pred_label,
                    actual=outcome,
                )

        # One consolidation check per batch: detectability + stability gated
        # (crystallizes only past the noise edge, never on a schedule).
        if self.enable_eigen_memory:
            self.kernel.check_and_crystallize()

        print(f"Batch Predictive Surprise (Avg NLL): {np.mean(predictive_surprises):.2f}")

    def _retrieve_context(self, vec):
        """Returns (context string, nearest retrieved embedding or None).

        The nearest embedding is what the residual (query - retrieved) is computed
        against — the contrastive pair the kernel consumes.
        """
        if not self.enable_retrieval:
            return "", None

        ctx_str = ""
        nn_vec = None
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT context_input, actual_outcome, surprise_score, embedding
                FROM episodic_buffer
                ORDER BY embedding <=> %s::vector
                LIMIT 3
            """, (vec.tolist(),))
            episodes = cur.fetchall()

            if episodes:
                ctx_str += "Similar Past Events:\n"
                for ep in episodes:
                    ctx_str += f"- Input: {ep[0]} -> Result: {ep[1]} (Surprise: {ep[2]:.2f})\n"
                nn_vec = np.array(json.loads(episodes[0][3]))

            # Only the Eigen arm injects crystallized axioms. Selection is by
            # |projection| of the centered query onto each axiom's axis — the old
            # cosine-to-eigenvector ORDER BY was sign-ambiguous and thus meaningless
            # (see THEORY.md section 4). Axiom counts are small, so scoring in
            # Python is fine.
            if self.enable_eigen_memory:
                cur.execute("SELECT axiom_content, eigen_vector FROM semantic_core")
                axioms = self.kernel.score_axioms(vec, cur.fetchall())[:AXIOM_INJECT_TOP_K]

                if axioms:
                    print(f"[AXIOM→] injected {len(axioms)} axiom(s) into context")
                    ctx_str += "\nRelevant Rules/Memories:\n"
                    for _, content in axioms:
                        ctx_str += f"- RULE: {content}\n"

        return (ctx_str if ctx_str else "No relevant memories found."), nn_vec

    def run(self, user_query):
        # Implementation of single-run legacy-compat if needed
        return self.run_batch([user_query])[0][0]
