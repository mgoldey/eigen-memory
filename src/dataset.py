import os
import random
import ssl
import urllib.request

# Valid label set per task. The agent and scorer read this instead of hardcoding
# RED/BLUE/GREEN, so a new task only needs an entry here plus a loader.
NUMBER_LABELS = ["RED", "BLUE", "GREEN"]
# TREC coarse classes; we use a 3-class subset to match the number-game's scale
# and keep the classes semantically distinct (so they separate in embedding space).
TREC_LABELS = ["HUM", "LOC", "NUM"]
# The C1-and-C3 polarity-flip routing task (docs/C1_C3_TASK.md).
FLIP_LABELS = ["ESCALATE", "FILE", "DEFER"]

LABELS = {
    "number": NUMBER_LABELS,
    "trec": TREC_LABELS,
    "flip": FLIP_LABELS,
}


def get_labels(task="number"):
    return LABELS[task]


def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def get_label(n):
    # Rule Priority:
    # 1. Prime -> RED
    # 2. Divisible by 5 -> BLUE
    # 3. Else -> GREEN
    if is_prime(n):
        return "RED"
    elif n % 5 == 0:
        return "BLUE"
    else:
        return "GREEN"

def generate_dataset(num_samples=100, seed=42):
    random.seed(seed)
    data = []
    # Generate a mix of numbers to ensure coverage of all classes
    # Range 1-100 to start simple
    for _ in range(num_samples):
        x = random.randint(1, 100)
        label = get_label(x)
        data.append({"input": x, "label": label})
    return data


# --- TREC question classification (short-text, embedding-friendly) ---------
# Unlike the number-game, the hidden rule (question type) is visible in text
# embedding space, so retrieval and PCA can actually exploit it. See
# docs/DATASETS.md and docs/USE_CASES.md.

_TREC_URLS = {
    "train": "https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label",
    "test": "https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label",
}
_TREC_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "eigen_memory_trec")


def _fetch_trec_raw(split):
    """Download (and cache) a raw TREC split file, returning its text."""
    os.makedirs(_TREC_CACHE, exist_ok=True)
    path = os.path.join(_TREC_CACHE, f"{split}.label")
    if not os.path.exists(path):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        raw = urllib.request.urlopen(_TREC_URLS[split], timeout=20, context=ctx).read()
        with open(path, "wb") as f:
            f.write(raw)
    with open(path, "rb") as f:
        return f.read().decode("latin-1")


def _parse_trec(text, keep_labels):
    """Parse 'COARSE:fine question' lines into {input, label}, keeping a subset."""
    rows = []
    for line in text.strip().split("\n"):
        label, _, question = line.partition(" ")
        coarse = label.split(":")[0]
        if coarse in keep_labels:
            rows.append({"input": question.strip(), "label": coarse})
    return rows


def load_trec(split="train", num_samples=100, seed=42, labels=None):
    """Load a class-subset of TREC, shuffled and truncated.

    split: 'train' or 'test' (test is the official held-out set — use it to
           measure generalization with memory frozen).
    """
    labels = labels or TREC_LABELS
    rows = _parse_trec(_fetch_trec_raw(split), set(labels))
    random.Random(seed).shuffle(rows)
    return rows[:num_samples]


# --- The polarity-flip routing task (C1-and-C3; docs/C1_C3_TASK.md) --------
# Short workplace messages with two hidden attributes: A = topic (dominant
# similarity axis) and B = speech act (request vs report; embedding-recoverable
# but sub-dominant). The hidden label FLIPS on B within each topic, so two
# same-topic messages get opposite labels: copying the nearest neighbor pays
# exactly the polarity-match rate m, and majority vote cannot exceed it. Only
# the abstract flip-rule generalizes. Train and test use DISJOINT subject
# phrases and frames (Guardrail 2: no shared surface vocabulary to memorize).

_FLIP_TOPICS = {
    # Compositional subject phrases: object x context, with DISJOINT train/test
    # banks for both (Guardrail 2). The combinatorial diversity (~48 train combos
    # per topic) keeps any single phrase near-unique in a 150-item buffer, so
    # retrieval cannot win by exact-phrase lookup and polarity cannot dominate
    # the neighbor choice (the m-tuning lever found via Guardrail 1).
    "billing": {
        "objects": {
            "train": ["the invoice", "the refund", "the credit memo",
                      "the purchase order", "the subscription renewal", "the payment plan"],
            "test": ["the chargeback", "the price adjustment"],
        },
        "contexts": {
            "train": ["for the Meridian account", "from the Hartley contract",
                      "on the March statement", "for Corvid Ltd",
                      "raised by the finance team", "tied to the Q3 renewal",
                      "for the Osprey account", "under the enterprise plan"],
            "test": ["for the Yardley account", "from the annual true-up",
                     "on the April statement"],
        },
    },
    "infrastructure": {
        "objects": {
            "train": ["the database migration", "the certificate rotation",
                      "the node pool upgrade", "the backup job",
                      "the failover config", "the log pipeline"],
            "test": ["the registry cleanup", "the replica lag fix"],
        },
        "contexts": {
            "train": ["on the staging cluster", "for the EU region",
                      "behind the API gateway", "on the build servers",
                      "for the payments service", "in the analytics stack",
                      "on the queue workers", "for the web tier"],
            "test": ["in the Frankfurt data center", "on the orders database",
                     "for the ingestion layer"],
        },
    },
    "hiring": {
        "objects": {
            "train": ["the onsite loop", "the offer letter", "the reference checks",
                      "the visa paperwork", "the debrief notes", "the headcount approval"],
            "test": ["the take-home review", "the relocation package"],
        },
        "contexts": {
            "train": ["for the backend candidate", "for the data engineering role",
                      "for the finance controller", "for the platform team",
                      "for the ML researcher", "from Thursday's panel",
                      "for the QA lead", "for the May onboarding class"],
            "test": ["for the mobile role", "for the operations manager",
                     "for the Zurich transfer"],
        },
    },
    "marketing": {
        "objects": {
            "train": ["the email sequence", "the booth design", "the case study",
                      "the nurture campaign", "the A/B test", "the launch deck"],
            "test": ["the sponsorship deal", "the landing pages"],
        },
        "contexts": {
            "train": ["for the spring launch", "for DevSummit", "with Bluepine Health",
                      "on the pricing page", "for the beta announcement",
                      "for the partner portal", "for the analyst briefing",
                      "in the newsletter"],
            "test": ["with Stack Signals", "for the LATAM push", "with Fernwood Labs"],
        },
    },
    "security": {
        "objects": {
            "train": ["the key rotation", "the access review", "the phishing simulation",
                      "the endpoint rollout", "the penetration test", "the dependency audit"],
            "test": ["the bug bounty triage", "the badge audit"],
        },
        "contexts": {
            "train": ["for the sales org", "in the identity provider",
                      "for contractor laptops", "from Calloway Bank",
                      "for the mobile app", "after the S3 incident",
                      "for the finance group", "in the monorepo"],
            "test": ["for the third floor", "in the checkout service", "for the new CRM"],
        },
    },
    "logistics": {
        "objects": {
            "train": ["the hardware shipment", "the customs paperwork",
                      "the inventory recount", "the freight quote",
                      "the damage claim", "the returns backlog"],
            "test": ["the fleet maintenance", "the import duties"],
        },
        "contexts": {
            "train": ["for the Austin office", "for the demo units",
                      "in the overflow warehouse", "for the trade-show gear",
                      "with the carrier", "from December",
                      "for the fourth floor", "on the Friday schedule"],
            "test": ["for the replacement routers", "for the prototype enclosures",
                     "on the loading dock"],
        },
    },
}

# Polarity as a small MARKER SLOT inside a compositional neutral shell
# (opener x tail): the polarity is carried by 2-3 words out of ~15, so it cannot
# dominate nearest-neighbor distance (C3), while the marker vocabulary is
# semantically crisp enough for a supervised probe to recover (C1). This shape
# was forced by Guardrail 1: whole-sentence polarity frames gave m = 0.71-0.86
# (retrieval pre-split on polarity) across two earlier designs. All banks have
# disjoint train/test splits (Guardrail 2).
_FLIP_DECOS = {
    "train": ["Quick status:", "Team note:", "Heads-up:", "Small update:", "Tracking note:"],
    "test": ["Log entry:", "Note for the channel:", "End-of-day note:"],
}
_FLIP_TAILS = {
    "train": ["per this morning's sync", "as of today", "for the current sprint",
              "on our side", "per the tracker"],
    "test": ["as of this afternoon", "per the weekly review", "going into next week"],
}
_FLIP_TAILS2 = {
    "train": ["flagged during standup", "logged in the tracker",
              "mentioned in the weekly notes", "copied to the leads",
              "captured in the minutes"],
    "test": ["noted in the retro", "shared in the channel", "added to the digest"],
}
# Marker pairs share their head word across polarity (handled/handled,
# review/review, ...): embedding models underweight negation and aspect, so the
# cross-polarity distance shrinks further, while the still/not-yet vs
# fully/already cues stay probe-recoverable.
_FLIP_MARKERS = {
    "request": {
        "train": ["not yet handled", "still awaiting review", "open as an action item",
                  "pending on our end", "waiting for an owner"],
        "test": ["not yet completed", "still in the queue", "unresolved for now"],
    },
    "report": {
        "train": ["fully handled", "already through review", "closed as an action item",
                  "resolved on our end", "handled by its owner"],
        "test": ["fully completed", "cleared from the queue", "resolved for good"],
    },
}


def flip_rule_table(seed=42):
    """The per-seed hidden rule: topic -> (label_for_request, label_for_report),
    always distinct within a topic (the flip)."""
    rng = random.Random(f"flip-lut-{seed}")
    return {t: tuple(rng.sample(FLIP_LABELS, 2)) for t in sorted(_FLIP_TOPICS)}


def flip_oracle_text(seed=42):
    """The rule rendered as text, for the Oracle_Rule ceiling arm."""
    lut = flip_rule_table(seed)
    lines = [
        "Routing rule: every message is either a REQUEST (asking for something to"
        " be done) or a REPORT (informing that something happened). Route by topic"
        " and message type:"
    ]
    for t, (req, rep) in lut.items():
        lines.append(f"- {t}: requests -> {req}; reports -> {rep}")
    return "\n".join(lines)


def load_flip(split="train", num_samples=100, seed=42):
    """Generate polarity-flip routing messages. The rule table depends only on
    the seed (same rule at train and test); the samples depend on seed+split.
    Each item carries meta (topic, polarity) for per-cell analysis."""
    lut = flip_rule_table(seed)
    rng = random.Random(f"flip-{split}-{seed}")
    topics = sorted(_FLIP_TOPICS)
    data = []
    for _ in range(num_samples):
        topic = rng.choice(topics)
        polarity = rng.choice(["request", "report"])
        banks = _FLIP_TOPICS[topic]
        phrase = f"{rng.choice(banks['objects'][split])} {rng.choice(banks['contexts'][split])}"
        deco = rng.choice(_FLIP_DECOS[split])
        tail = rng.choice(_FLIP_TAILS[split])
        tail2 = rng.choice(_FLIP_TAILS2[split])
        marker = rng.choice(_FLIP_MARKERS[polarity][split])
        msg = f"{deco} {phrase} is {marker}, {tail} - {tail2}."
        label = lut[topic][0] if polarity == "request" else lut[topic][1]
        data.append({"input": msg, "label": label,
                     "meta": {"topic": topic, "polarity": polarity}})
    return data


def load_dataset(task="number", split="train", num_samples=100, seed=42):
    """Unified entry point. task='number' (default), 'trec', or 'flip'."""
    if task == "number":
        return generate_dataset(num_samples=num_samples, seed=seed)
    if task == "trec":
        return load_trec(split=split, num_samples=num_samples, seed=seed)
    if task == "flip":
        return load_flip(split=split, num_samples=num_samples, seed=seed)
    raise ValueError(f"Unknown task: {task!r}")


if __name__ == "__main__":
    print("number:", generate_dataset(3))
    print("trec:", load_trec(num_samples=3))
