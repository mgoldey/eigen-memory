# Archived: the compromised first multi-seed flip run

These are **not** current results. They are kept deliberately, as the record of a run whose
numbers were corrupted by two silent bugs found in a later review pass:

1. The crystallizer stored the model's full chain-of-thought as the "axiom" and injected
   ~1.2k characters of it into every Treatment context — sabotaging the arm under test.
2. The surprise probe's token matching failed on multi-token labels, flattening the
   prediction-error signal to a constant for 2 of 3 classes. This was the *third* instance
   of the constant/corrupted-signal bug class this project kept hitting (see docs/FINDINGS.md for
   the running count).

Both were verified live before being fixed (NLLs 7.0 / 0.01 / 7.0 → 4.57 / 0.01 / 11.55 after
a one-line prefix match). The corrected run that replaced these files lives in
[`../flip/`](../flip/).

Full account: [`../../docs/FINDINGS.md`](../../docs/FINDINGS.md) (§ "The third act") and
[`../../docs/BLOG_POST.md`](../../docs/BLOG_POST.md).
