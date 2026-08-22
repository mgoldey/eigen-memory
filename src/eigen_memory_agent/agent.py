import json
import math
import re

import numpy as np
from openai import OpenAI
import psycopg2

from .memory_kernel import EigenMemoryKernel, KernelConfig
from .parsing import parse_prediction, clean_prediction

try:
    from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL
except ImportError:  # allow import when run as a standalone module
    OLLAMA_BASE_URL = "http://localhost:11434/v1"
    EMBEDDING_MODEL = "embeddinggemma:latest"

# --- Surprise / memory thresholds (in nats of NLL) ---
# Surprise assigned when the true-label token is absent from the top-k logprobs.
# Floors a "missing" token at a high-but-finite value so it reliably counts as
# surprising without the old bug where this case silently collapsed to a flat 2.0.
MISSING_TOKEN_NLL = 7.0
# Cap on the prediction call's chain-of-thought tokens. Enough room for a short
# <thought> block plus the final label line, without the ~1.3k-char ramble that
# makes each call slow and tends to make a 4B model drift.
PREDICTION_MAX_TOKENS = 200
# How many axioms to inject into context (Treatment arm only).
AXIOM_INJECT_TOP_K = 2
# Fixed sampling seed for every LLM call: per-run variation must come from the
# DATA seed, not from Ollama's default temperature=0.8 sampling noise.
LLM_SEED = 0

# The prediction prompt, module-level so it is reviewable next to the surprise
# probe (and so its lines carry no accidental indentation into the model).
PREDICTION_PROMPT = """\
You are playing a pattern recognition game.
Valid outputs are only: {label_str}.

{context}

Task: Analyze the input and rules inside <thought> blocks.
Your final line must be exactly one of: {label_str}.

Input: {x}
Output:"""


def _extract_nll(top_logprobs, true_label):
    """Negative log-likelihood (nats) of the true label among top-k logprobs.

    The probe emits at most one token, and tokenizers split long labels
    ("ESCALATE" -> "ES"/"ESC", "DEFER" -> "DE"), so the match is: the candidate
    token must be a PREFIX of the true label. The previous check required the
    full label inside one token, which silently returned MISSING_TOKEN_NLL for
    every item of a multi-token class — the third instance of this repo's
    constant-surprise bug class (see docs/FINDINGS.md). Label sets must therefore
    have distinct first tokens (RED/BLUE/GREEN, HUM/LOC/NUM,
    ESCALATE/FILE/DEFER all do).

    Returns MISSING_TOKEN_NLL when no token matches, rather than leaving the
    value undefined (the original bug).
    """
    lab = true_label.strip().upper()
    for tl in top_logprobs:
        tok = tl.token.upper().strip(" .,:;!?'\"")
        if tok and lab.startswith(tok):
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




class AgenticMemoryLoop:
    """The agent loop: embed, retrieve, predict, measure surprise, learn.

    Arms are configured by the keyword flags: retrieval off + eigen off is the
    Baseline; retrieval on is Control_RAG; both on is Treatment_Eigen; a
    static_context with retrieval off is an Oracle arm.
    """

    def __init__(self, db_string=None, openai_client=None, *, db_conn=None,
                 model="gemma3:4b",
                 thought_model="gemma3:4b", enable_retrieval=True,
                 enable_eigen_memory=True, labels=None, static_context="",
                 extra_body=None, retrieval_k=3, recency_rerank=False,
                 axiom_replaces_exemplars=False, kernel_kwargs=None):
        if db_conn is not None:
            self.conn = db_conn
        elif db_string is not None:
            self.conn = psycopg2.connect(db_string)
        else:
            raise TypeError("provide db_string or db_conn")
        # Valid label set for this task (RED/BLUE/GREEN by default; e.g.
        # HUM/LOC/NUM for TREC). Used in both the prediction and surprise prompts.
        self.labels = labels or DEFAULT_LABELS
        # Fixed context prepended to every prediction (e.g. the true rule for an
        # Oracle_Rule ceiling arm). Empty for normal arms.
        self.static_context = static_context
        # If client not provided, default to local Ollama
        if openai_client is None:
            # Explicit timeout + retries. The SDK default is 600 s, so a single
            # dropped keepalive to a LOCAL server stalls a 2050-call run for ten
            # minutes (observed twice on 2026-08-11: socket ESTAB, 0 bytes queued
            # both directions, while a fresh request to the same server answered
            # in under 2 s). Calls here take ~9 s, so 120 s is generous headroom
            # while still failing fast enough to retry rather than hang.
            self.client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama",
                                 timeout=120.0, max_retries=3)
        else:
            self.client = openai_client

        self.model = model
        self.thought_model = thought_model
        self.enable_retrieval = enable_retrieval
        self.enable_eigen_memory = enable_eigen_memory
        # Extra request-body fields for every chat call (e.g.
        # {"reasoning_effort": "none"}: on Ollama's OpenAI-compat endpoint a
        # thinking-family model otherwise burns the whole completion budget on
        # a reasoning stream and returns EMPTY content — RFμ bug five).
        self.extra_body = extra_body
        # Retrieval policy knobs for the Rule-Shift arms: top-k episodes,
        # whether to present them newest-first (the Recency_RAG kill arm), and
        # the pre-registered Treatment injection policy — a selected axiom
        # REPLACES the exemplar block instead of being appended to it.
        self.retrieval_k = retrieval_k
        self.recency_rerank = recency_rerank
        self.axiom_replaces_exemplars = axiom_replaces_exemplars
        # labels reaches the kernel so axiom validation can floor its accept bar
        # at chance (§9b: a rule passed at 0.30 vs a 0.20 baseline on a 3-label
        # task, both below the 0.33 chance line). An explicit kernel_kwargs
        # value still wins.
        _kkw = {"labels": self.labels, **(kernel_kwargs or {})}
        self.kernel = EigenMemoryKernel(self.conn, self.client, model=self.thought_model,
                                        extra_body=extra_body, **_kkw)
        # An episode is written when its predictive surprise clears chance-level
        # NLL for this label set (ln 3 ≈ 1.10 for three classes). The old fixed
        # 1.0 threshold sat BELOW chance, making the gate nearly write-everything.
        self.write_nll = math.log(len(self.labels))
        # Health counters, persisted into results by the experiment scripts so a
        # degraded signal is loud instead of silently constant (see docs/FINDINGS.md).
        self.embed_failures = 0
        self.nll_probes = 0
        self.nll_missing = 0
        self.logprob_missing = 0

    def _embed(self, text):
        try:
            res = self.client.embeddings.create(input=str(text), model=EMBEDDING_MODEL)
            return np.array(res.data[0].embedding)
        except Exception as e:
            # No fallback vector: a random embedding written to the buffer would
            # corrupt retrieval for the rest of the run. Skip memory for this
            # item instead, and count it so a degraded run is visible.
            self.embed_failures += 1
            print(f"Warning: embedding failed ({e}); skipping memory for this "
                  f"item (total failures: {self.embed_failures})")
            return None


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
            context, nn_vec = ("", None) if q_vec is None else self._retrieve_context(q_vec)
            if self.static_context:
                context = self.static_context + ("\n\n" + context if context else "")
            contexts.append(context)
            residuals.append(q_vec - nn_vec if nn_vec is not None else None)

            # 2. Predict with CoT and Logprobs
            label_str = ", ".join(self.labels)
            prompt = PREDICTION_PROMPT.format(label_str=label_str, context=context, x=x)

            try:
                pred_resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    logprobs=True,
                    top_logprobs=5,  # To see distribution across labels
                    temperature=0.0,
                    seed=LLM_SEED,
                    # Cap the chain-of-thought. Uncapped, gemma3:4b emits ~1.3k chars
                    # of reasoning per call (slow, and long CoT tends to drift on a
                    # 4B model). A short budget keeps the final RED/BLUE/GREEN line.
                    max_tokens=PREDICTION_MAX_TOKENS,
                    extra_body=self.extra_body,
                )
                raw = pred_resp.choices[0].message.content
                predictions.append(raw)

                # Calculate Perceptual Surprise (Entropy of the first token)
                if pred_resp.choices[0].logprobs and pred_resp.choices[0].logprobs.content:
                    logprobs_info = pred_resp.choices[0].logprobs.content
                    top = logprobs_info[0].top_logprobs
                    probs = [math.exp(tl.logprob) for tl in top]
                    entropy = -sum(p * math.log(p) for p in probs)
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
                    max_tokens=1,
                    temperature=0.0,
                    seed=LLM_SEED,
                    extra_body=self.extra_body,
                )

                if res.choices[0].logprobs and res.choices[0].logprobs.content:
                    lp_content = res.choices[0].logprobs.content[0]
                    # NLL of the true label among top-k. If the true-label token is
                    # absent from top-k, _extract_nll returns a high, finite value
                    # (the original code left target_lp undefined here and silently
                    # collapsed every miss to a flat 2.0 via the except path).
                    s_pred = _extract_nll(lp_content.top_logprobs, outcome)
                    self.nll_probes += 1
                    if s_pred == MISSING_TOKEN_NLL:
                        self.nll_missing += 1
                else:
                    # No logprobs returned — skip the write rather than
                    # fabricating a surprise score from uncalibrated proxies.
                    s_pred = 0.0
                    self.logprob_missing += 1

                predictive_surprises.append(s_pred)

            except Exception as e:
                print(f"Surprise Eval Error: {e}")
                s_pred = 0.0  # skip the write — fabricating surprise hides bugs
                predictive_surprises.append(s_pred)

            # Store in Episodic Buffer if either Salient (Perceptual) or Surprising
            # (Predictive). Skipped when embedding failed — a memory row without a
            # real embedding poisons retrieval.
            if q_vec is not None and (is_salient or s_pred > self.write_nll):
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
            # Retire before crystallizing: a rule that just went stale should not
            # be competing for injection alongside its replacement, and freeing
            # its direction lets the new one crystallize on the same axis.
            self.kernel.revalidate_axioms()
            self.kernel.check_and_crystallize()

        print(f"Batch Predictive Surprise (Avg NLL): {np.mean(predictive_surprises):.2f}")

    def _retrieve_context(self, vec):
        """Returns (context string, nearest retrieved embedding or None).

        The nearest embedding is what the residual (query - retrieved) is computed
        against — the contrastive pair the kernel consumes.
        """
        if not self.enable_retrieval:
            return "", None

        episode_str = ""
        axiom_str = ""
        nn_vec = None
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT context_input, actual_outcome, surprise_score, embedding, created_at
                FROM episodic_buffer
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (vec.tolist(), self.retrieval_k))
            episodes = cur.fetchall()

            if episodes:
                # The residual is always against the nearest-by-similarity
                # episode, regardless of how the context is presented.
                nn_vec = np.array(json.loads(episodes[0][3]))
                if self.recency_rerank:
                    shown = sorted(episodes, key=lambda ep: ep[4], reverse=True)
                    episode_str += ("Similar Past Events (most recent first — "
                                    "labels may have changed over time):\n")
                else:
                    shown = episodes
                    episode_str += "Similar Past Events:\n"
                for ep in shown:
                    episode_str += f"- Input: {ep[0]} -> Result: {ep[1]} (Surprise: {ep[2]:.2f})\n"

            # Only the Eigen arm injects crystallized axioms. Selection is by
            # |projection| of the centered query onto each axiom's axis — the old
            # cosine-to-eigenvector ORDER BY was sign-ambiguous and thus meaningless
            # (see THEORY.md section 4). Axiom counts are small, so scoring in
            # Python is fine.
            if self.enable_eigen_memory:
                cur.execute(self.kernel.axiom_select_sql())
                axioms = self.kernel.score_axioms(vec, cur.fetchall())[:AXIOM_INJECT_TOP_K]

                if axioms:
                    print(f"[AXIOM→] injected {len(axioms)} axiom(s) into context")
                    axiom_str += "\nRelevant Rules/Memories:\n"
                    for _, content in axioms:
                        # content is the stored "RULE: ..." line (the kernel
                        # strips the CoT scaffolding before the INSERT).
                        axiom_str += f"- {content}\n"

        if axiom_str and self.axiom_replaces_exemplars:
            # Pre-registered Rule-Shift injection policy: once a rule is
            # trusted enough to inject, it REPLACES the exemplars — mixing a
            # current rule with stale exemplars is exactly the RFμ RC failure.
            episode_str = ""
        ctx_str = episode_str + axiom_str

        return (ctx_str if ctx_str else "No relevant memories found."), nn_vec

    def run(self, user_query):
        return self.run_batch([user_query])[0][0]
