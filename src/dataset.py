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

LABELS = {
    "number": NUMBER_LABELS,
    "trec": TREC_LABELS,
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


def load_dataset(task="number", split="train", num_samples=100, seed=42):
    """Unified entry point. task='number' (default) or 'trec'."""
    if task == "number":
        return generate_dataset(num_samples=num_samples, seed=seed)
    if task == "trec":
        return load_trec(split=split, num_samples=num_samples, seed=seed)
    raise ValueError(f"Unknown task: {task!r}")


if __name__ == "__main__":
    print("number:", generate_dataset(3))
    print("trec:", load_trec(num_samples=3))
