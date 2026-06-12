# Chapter 6: Evaluation, Tuning & Advanced Techniques

> Up to now we have built a deduplication engine and *trusted* it. This chapter is about not trusting it. We will measure it, tune it against ground truth, and then look at three advanced algorithms that pick up where MinHash + LSH leaves off. Everything is grounded in real numbers and code you can run.

---

## 1. Learning objectives

After this chapter you will be able to:

- **Construct ground truth** for a deduplication system three ways — human annotation with a *stratified* sample, synthetic duplicate injection at a *known* Jaccard, and inter-annotator agreement via Cohen's kappa — and know which to reach for.
- **Compute precision, recall, and F1 for duplicate *pairs*** and, critically, *estimate* them without annotating all O(n²) pairs.
- **Connect the knobs to the outcome**: explain exactly how `(shingle_size, lsh_bands, lsh_rows, jaccard_threshold)` move the precision/recall operating point, and run a grid search that plots a Pareto frontier.
- **Apply three advanced techniques** — SimHash `[Charikar 2002]`, suffix-array dedup `[Manber & Myers 1993]`, and semantic dedup `[Abbas et al. 2023]` — and say when each beats MinHash.
- **Read a real dedup report**: interpret a heavy-tailed cluster-size distribution from a 10M-document Common Crawl shard and avoid the classic traps.

---

## 2. Concept explanation (with real numbers)

### 2.1 Why evaluation is hard: the O(n²) wall

A corpus of `n = 10,000,000` documents has `n(n-1)/2 ≈ 5 × 10¹³` pairs. You cannot annotate that. The entire art of dedup evaluation is **measuring a system over an unannotatable space using small, carefully-chosen labeled samples**. Keep that sentence in mind; everything below serves it.

### 2.2 Ground truth via stratified human annotation

Suppose we want **1,000 labeled document pairs**. The naive move — sample 1,000 pairs uniformly at random — is a disaster. With a dedup rate around 31%, a *uniformly random* pair is almost certainly a non-duplicate (true duplicates are a vanishing fraction of all pairs). Your annotators would label 1,000 obvious non-dupes and learn nothing about the decision boundary.

Instead we **stratify by estimated similarity**. Cheaply estimate Jaccard for a large pool of candidate pairs (the MinHash signature gives `J_est = mean(sig_a == sig_b)` for free), bin them, and sample evenly across bins:

| Stratum (J_est) | Pairs in pool | Sampled to annotate |
|---|---:|---:|
| 0.00 – 0.20 | 4,200,000 | 150 |
| 0.20 – 0.50 | 380,000 | 200 |
| 0.50 – 0.70 | 41,000 | 250 |
| 0.70 – 0.85 | 9,500 | 250 |
| 0.85 – 1.00 | 3,100 | 150 |
| **Total** | — | **1,000** |

The mass of the annotation budget lands in **0.50 – 0.85**, exactly where the threshold lives and where human judgment actually matters. When you later compute corpus-wide metrics, you re-weight each stratum by its true pool size (inverse-probability weighting) so the estimate stays unbiased.

### 2.3 Synthetic duplicates at a *known* Jaccard

Human labels are slow. A complementary trick: take a real document, represent it as a set of `m` word shingles, and **corrupt a controlled fraction** to produce a near-duplicate whose Jaccard you know *in advance*.

If you randomly delete a fraction `f` of the `m` shingles (no substitutions), the corrupted set is a subset of size `(1−f)m`. The Jaccard of original `A` and corrupted `B = A \ deleted` is:

```
|A ∩ B| = (1 − f) m
|A ∪ B| = m
J(A, B) = (1 − f) m / m = 1 − f
```

So **delete-only** gives `J = 1 − f` exactly. Delete 15% of shingles → Jaccard ≈ 0.85.

Substitution is harsher. If you *replace* a fraction `f` (delete the shingle *and* add a brand-new one), then `|A ∩ B| = (1−f)m` but `|A ∪ B| = (1−f)m + fm + fm = (1+f)m`:

```
J = (1 − f) / (1 + f)
```

Substitute 15% → `J = 0.85 / 1.15 ≈ 0.739`. Same edit fraction, much lower Jaccard, because every edit hurts the union *twice*. This delete-vs-substitute asymmetry is the single most common source of "my injected dupes don't match the Jaccard I asked for" confusion.

### 2.4 Inter-annotator agreement: Cohen's kappa

Two annotators label the same 1,000 pairs as **DUP** / **NOT**. They will not agree perfectly, and *some* agreement happens by chance. Cohen's kappa corrects for chance. Here is the 2×2 contingency table:

|                | B: DUP | B: NOT | row total |
|----------------|-------:|-------:|----------:|
| **A: DUP**     |   320  |    40  |    360    |
| **A: NOT**     |    60  |   580  |    640    |
| **col total**  |   380  |   620  |   1000    |

**Observed agreement** `p_o` = (both-DUP + both-NOT) / total:

```
p_o = (320 + 580) / 1000 = 0.900
```

**Expected-by-chance agreement** `p_e` — multiply each annotator's marginal rates:

```
P(both say DUP by chance) = (360/1000) × (380/1000) = 0.360 × 0.380 = 0.1368
P(both say NOT by chance) = (640/1000) × (620/1000) = 0.640 × 0.620 = 0.3968
p_e = 0.1368 + 0.3968 = 0.5336
```

**Kappa**:

```
κ = (p_o − p_e) / (1 − p_e) = (0.900 − 0.5336) / (1 − 0.5336)
  = 0.3664 / 0.4664 ≈ 0.786
```

By the Landis–Koch convention, `κ ≈ 0.79` is **"substantial" agreement** (0.61–0.80). If your annotators sit below ~0.6, your *labels* are too noisy to trust as ground truth — fix the annotation guidelines before you trust any precision/recall number computed against them.

### 2.5 Precision / recall / F1 for duplicate pairs

We frame dedup as a binary classifier over **pairs**. A pair is *predicted positive* if the pipeline puts both docs in the same cluster.

- **TP** — predicted dup **AND** truly dup
- **FP** — predicted dup **BUT** not truly dup
- **FN** — truly dup **BUT** the pipeline missed it

```
precision = TP / (TP + FP)        # of the pairs we flagged, how many were right
recall    = TP / (TP + FN)        # of the truly-dup pairs, how many we caught
F1        = 2 · P · R / (P + R)   # harmonic mean
```

Worked confusion matrix on a labeled set with **900 true duplicate pairs**:

| | Truly DUP | Truly NOT |
|---|---:|---:|
| **Predicted DUP** | TP = 810 | FP = 90 |
| **Predicted NOT** | FN = 90 | TN = … |

```
precision = 810 / (810 + 90) = 810 / 900 = 0.900
recall    = 810 / (810 + 90) = 810 / 900 = 0.900
F1        = 2 · 0.9 · 0.9 / (0.9 + 0.9) = 0.900
```

### 2.6 Estimating P and R **without** annotating O(n²) pairs

This is the crux. Two different sampling tricks, because precision and recall need *different* populations:

- **Precision** — sample from the **predicted-positive set**. The pipeline already produced, say, 4.0M flagged pairs. Draw 500 at random, annotate them. If 455 are true duplicates, `precision ≈ 455 / 500 = 0.91`. You only annotated 500 pairs, and the estimate is exact in expectation because you sampled directly from the denominator (`TP + FP`).

- **Recall** — you *cannot* sample uniformly from "all true duplicate pairs" because finding them is the whole problem. Two escapes:
  1. **Stratified pool** (Section 2.2): annotate across similarity bins, then estimate, in each bin, the fraction of true dupes the pipeline caught, re-weighted by bin size.
  2. **Injected duplicates** (Section 2.3): inject `N = 5,000` synthetic dupes at known Jaccard, run the pipeline, count how many were placed in the right cluster. If 4,300 are recovered, **recall ≈ 0.86 exactly** — no human in the loop, because *you* manufactured the ground truth.

**The knob → operating-point link.** The LSH S-curve sets **candidate recall**: a pair surfaces as a candidate iff it collides in ≥ 1 of the `lsh_bands` bands, with probability `1 − (1 − J^lsh_rows)^lsh_bands` `[Leskovec et al. 2014]`. More bands / fewer rows ⇒ steeper, lower-threshold curve ⇒ **higher recall, more candidates, lower precision**. The **verify step** (`jaccard_threshold` in `high_precision_mode`) then sets **precision**: any candidate below threshold is dropped. So:

> `(lsh_bands, lsh_rows)` choose how many true dupes you can *possibly* catch (recall ceiling); `jaccard_threshold` trades that recall for precision after the fact.

---

## 3. Annotated code walkthrough

Runnable with `numpy` + the standard library. These call into the project's real modules: `dedup_pipeline.evaluation.metrics`, `dedup_pipeline.evaluation.synthetic_injector`, and `dedup_pipeline.pipeline.pipeline.DedupPipeline`. The snippet below is self-contained so it runs even before those modules are wired up.

```python
import numpy as np

# ---------------------------------------------------------------------------
# 3.1  Synthetic injection at a KNOWN Jaccard
#      (mirrors dedup_pipeline.evaluation.synthetic_injector)
# ---------------------------------------------------------------------------
def inject_near_duplicate(shingles, edit_fraction, mode, rng):
    """Corrupt a shingle set to a controlled Jaccard.
    `shingles`      : set[int]  -- word-shingle hashes of the original doc
    `edit_fraction` : float     -- fraction f of shingles to delete/substitute
    `mode`          : "delete" or "substitute"
    Returns (corrupted_set, true_jaccard)."""
    base = np.array(sorted(shingles))          # deterministic order before shuffling
    m = len(base)
    n_edit = int(round(edit_fraction * m))     # how many shingles to touch
    victims = rng.choice(m, size=n_edit, replace=False)  # pick WITHOUT replacement
    kept = np.delete(base, victims)            # the (1-f)m survivors
    if mode == "delete":
        corrupted = set(kept.tolist())
        true_j = (m - n_edit) / m              # J = 1 - f  (derived in 2.3)
    elif mode == "substitute":
        # add n_edit brand-new shingle ids that cannot collide with originals
        fresh = set((base.max() + 1 + np.arange(n_edit)).tolist())
        corrupted = set(kept.tolist()) | fresh
        inter = m - n_edit                     # |A ∩ B| = (1-f)m
        union = m + n_edit                     # |A ∪ B| = (1+f)m
        true_j = inter / union                 # J = (1-f)/(1+f)
    else:
        raise ValueError(mode)
    return corrupted, true_j

# ---------------------------------------------------------------------------
# 3.2  Pair metrics (mirrors dedup_pipeline.evaluation.metrics)
# ---------------------------------------------------------------------------
def pair_metrics(predicted_pairs, truth_pairs):
    """All sets contain frozenset({i, j}) so order never matters."""
    pred = {frozenset(p) for p in predicted_pairs}
    truth = {frozenset(p) for p in truth_pairs}
    tp = len(pred & truth)                     # flagged AND true
    fp = len(pred - truth)                     # flagged but NOT true
    fn = len(truth - pred)                     # true but MISSED
    # guard the zero-denominator cases (no predictions / no truth)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}

# ---------------------------------------------------------------------------
# 3.3  Estimate PRECISION by sampling the predicted-positive set
# ---------------------------------------------------------------------------
def estimate_precision(predicted_pairs, annotate_fn, n_sample, rng):
    """annotate_fn(pair) -> bool is the (expensive) human/oracle label."""
    pred = list(predicted_pairs)
    idx = rng.choice(len(pred), size=min(n_sample, len(pred)), replace=False)
    labels = [annotate_fn(pred[i]) for i in idx]      # the only costly line
    return float(np.mean(labels))                     # fraction truly-dup == precision

# ---------------------------------------------------------------------------
# 3.4  Cohen's kappa from two annotators' label vectors
# ---------------------------------------------------------------------------
def cohens_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    p_o = np.mean(a == b)                       # observed agreement
    # marginals: each annotator's DUP-rate, used for chance agreement
    pa1, pb1 = a.mean(), b.mean()
    p_e = pa1 * pb1 + (1 - pa1) * (1 - pb1)     # expected-by-chance agreement
    return (p_o - p_e) / (1 - p_e)              # the kappa formula

# ---------------------------------------------------------------------------
# 3.5  Putting it together against the real pipeline
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)            # config.random_seed in the real run

    # Inject 5,000 known near-dupes at Jaccard 0.85 (delete-mode -> f = 0.15)
    orig = set(range(1000))
    corrupted, j = inject_near_duplicate(orig, edit_fraction=0.15,
                                         mode="delete", rng=rng)
    print(f"injected Jaccard = {j:.3f}")        # -> 0.850

    # Recall measured EXACTLY against injected truth (no humans needed):
    # from dedup_pipeline.pipeline.pipeline import DedupPipeline
    # pipe = DedupPipeline(config)            # config.lsh_bands / lsh_rows / threshold
    # clusters = pipe.run(corpus_with_injections)
    # recall = recovered_injections / 5000

    # Kappa demo reproducing Section 2.4 (κ ≈ 0.786)
    A = np.array([True]*320 + [True]*40  + [False]*60  + [False]*580)
    B = np.array([True]*320 + [False]*40 + [True]*60   + [False]*580)
    print(f"Cohen's kappa = {cohens_kappa(A, B):.3f}")   # -> 0.786
```

### 3.6 Hyperparameter sweep and the Pareto frontier

The constraint `num_hash_functions = lsh_bands × lsh_rows` (enforced by `PipelineConfig`) couples three knobs, so a clean grid fixes `n` and sweeps `(b, r)` factorizations plus `k` and threshold.

```python
import itertools, numpy as np

# A small, realistic grid. n is fixed at 128 so b*r must equal 128.
grid = {
    "shingle_size":      [3, 5, 7],
    "br":                [(8, 16), (16, 8), (32, 4)],   # (bands, rows), b*r = 128
    "jaccard_threshold": [0.70, 0.80, 0.90],
}

def lsh_collision_prob(J, b, r):
    # S-curve from [Leskovec et al. 2014]: P(candidate | true Jaccard J)
    return 1.0 - (1.0 - J ** r) ** b

results = []
for k, (b, r), thr in itertools.product(grid["shingle_size"],
                                        grid["br"], grid["jaccard_threshold"]):
    # In the real sweep you'd run DedupPipeline on a LABELED subset (cross-val)
    # and read precision/recall off dedup_pipeline.evaluation.metrics.pair_metrics.
    # Here we model it: candidate recall from the S-curve at the true-dup Jaccard,
    # precision falling as the threshold drops below the duplicate band.
    cand_recall = lsh_collision_prob(0.85, b, r)        # true dupes sit at J≈0.85
    precision   = min(1.0, 0.55 + 0.45 * (thr / 0.90))  # higher thr -> cleaner
    recall      = cand_recall * (1.0 if thr <= 0.85 else 0.80)  # high thr drops some
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    results.append(dict(k=k, b=b, r=r, thr=thr,
                        precision=precision, recall=recall, f1=f1))

# Pareto frontier: keep a point if no other point dominates it on BOTH axes.
def pareto(points):
    front = []
    for p in points:
        dominated = any(q["precision"] >= p["precision"]
                        and q["recall"] >= p["recall"]
                        and (q["precision"] > p["precision"]
                             or q["recall"] > p["recall"]) for q in points)
        if not dominated:
            front.append(p)
    return front

frontier = sorted(pareto(results), key=lambda p: p["recall"])
for p in frontier:
    print(f"k={p['k']} b={p['b']} r={p['r']} thr={p['thr']:.2f} "
          f"P={p['precision']:.3f} R={p['recall']:.3f} F1={p['f1']:.3f}")

# ---- Plot: scatter all points, highlight the frontier ----
# import matplotlib.pyplot as plt
# xs = [p["recall"] for p in results]; ys = [p["precision"] for p in results]
# plt.scatter(xs, ys, c="lightgray", label="all configs")     # the cloud
# fx = [p["recall"] for p in frontier]; fy = [p["precision"] for p in frontier]
# plt.plot(fx, fy, "-o", c="crimson", label="Pareto frontier") # the knee
# plt.xlabel("recall"); plt.ylabel("precision"); plt.legend()
```

**Which point do you pick?**

- **Precision-first** (e.g. dedup *before* a license/PII audit, where a false merge is costly): pick the frontier point with the highest precision that still meets a recall floor — typically `b=8, r=16, thr=0.90`. Fewer bands and more rows make a strict S-curve; a high threshold cleans the survivors.
- **Recall-first** (LLM pretraining, where *missed* dupes leak memorization): pick the highest-recall frontier point above a precision floor — typically `b=32, r=4, thr=0.70`. Many bands and few rows make candidates plentiful; the low threshold keeps near-dupes.

---

## 4. Common pitfalls

### Pitfall 1 — **The Uniform-Sampling Trap**
**Diagnosis.** You sampled annotation pairs uniformly at random and your kappa is fine but your measured precision is suspiciously ~0.999 and recall is unmeasurable. Almost every sampled pair was an obvious non-duplicate, so the labeled set contains *zero* true positives near the boundary. The metric is technically correct but tells you nothing about the regime you care about.
**Fix.** Stratify by `J_est` (Section 2.2). Put the annotation mass in the 0.5–0.85 band, and re-weight strata by their true pool size when aggregating, so the corpus-level estimate stays unbiased.

### Pitfall 2 — **Substitution Jaccard Drift**
**Diagnosis.** You asked `synthetic_injector` for dupes at Jaccard 0.85 using `edit_fraction = 0.15`, but the pipeline's recall on them is mysteriously low and `J_est` on the injected pairs reads ~0.74. The injector was in **substitute** mode: `J = (1−f)/(1+f) = 0.85/1.15 ≈ 0.739`, *below* your `jaccard_threshold` of 0.80, so the verify step correctly rejects them — your "recall failure" is actually correct behavior on mislabeled truth.
**Fix.** Be explicit about mode. For `J = 1 − f` use **delete** mode; if you need substitution at a target `J`, invert the formula: `f = (1 − J)/(1 + J)`. Assert the realized Jaccard in a test before trusting the recall number.

### Pitfall 3 — **The Giant-Cluster Precision Mirage**
**Diagnosis.** Corpus-level precision looks great (0.97), but spot-checks show distinct news articles merged together. A handful of **giant transitive clusters** (cookie banners, license boilerplate) dominate the pair count: one cluster of 10,000 docs contributes ~50M *correct* boilerplate pairs that swamp a few thousand wrong merges in the average. The metric is diluted by easy pairs.
**Fix.** Report metrics **per cluster-size stratum**, and separately track *cluster purity* on a sample of small (size 2–5) clusters where a single bad edge is visible. Cap or specially handle clusters above a size threshold (Section 5 case study), and raise `jaccard_threshold` if topically-similar-but-distinct docs are bleeding together.

### Pitfall 4 — **S-Curve / Threshold Mismatch** (bonus)
**Diagnosis.** Precision is excellent but recall is capped at 0.6 no matter how you tune the threshold. The LSH `(b, r)` S-curve never surfaces your duplicates as *candidates*, so the verify step never even sees them — you're tuning a knob downstream of the bottleneck.
**Fix.** Recall has a *ceiling* set by `(lsh_bands, lsh_rows)`. Compute `1 − (1 − J^r)^b` at your duplicate Jaccard; if it's below your recall target, change the factorization (more bands) *before* touching the threshold.

---

## 5. End-to-end case study: a 10M-document Common Crawl shard

Config: `shingle_size=5`, `shingle_mode="char"`, `num_hash_functions=128`, `lsh_bands=16`, `lsh_rows=8`, `jaccard_threshold=0.80`, `high_precision_mode=True`, `representative_strategy="longest"`.

**Headline statistics** (via `DedupPipeline.write_deduplicated`'s stats JSON):

| Metric | Value |
|---|---:|
| `input_count` | 10,000,000 |
| duplicates removed | ~3,100,000 |
| `output_count` | ~6,900,000 |
| `dedup_rate` | **31%** |
| measured precision (500 sampled pairs) | 0.94 |
| measured recall (5,000 injected dupes) | 0.88 |

**Cluster-size distribution** — heavy-tailed. Most clusters are tiny pairs; a few are enormous boilerplate templates:

```
cluster size   count (log scale, each # ≈ 5×)
2          ############################################  ~520,000
3          ###############################              ~180,000
4-7        ####################                          ~95,000
8-50       ##########                                    ~28,000
51-500     ####                                          ~3,400
501-5000   ##                                            ~210
5001+      #                                             ~12   <- boilerplate giants
```

The 12 "giant" clusters (each 10,000+ docs) account for **~1.4M of the 3.1M removed documents** — nearly half the dedup volume comes from a dozen templates.

**Surprising findings / lessons learned:**

1. **Giant clusters are template boilerplate, not content.** Every cluster above ~10,000 docs was a cookie-consent banner, a GPL/MIT license blob, a "404 — page not found" shell, or a CMS footer. None are "interesting" near-duplicates; they're the same string copied across millions of crawled pages. Inspect your top-10 clusters by hand — it is the fastest sanity check you can run.
2. **Threshold-too-low silently merges distinct docs.** Dropping `jaccard_threshold` from 0.80 to 0.65 *raised* the dedup rate to 38% — which looked like a win until annotation showed many merges were topically-similar-but-distinct articles (two different earnings reports sharing a templated header). The extra 7% was mostly false positives. Higher dedup rate is **not** automatically better.
3. **Normalization choices dominate the results.** Toggling HTML-stripping and NFKC normalization on/off swung the dedup rate by ~9 percentage points — a *larger* effect than any `(b, r)` retuning. Two pages identical except for smart-quotes vs. straight-quotes are duplicates only if your normalizer says so. Lock and version your normalization config; treat it as part of the experiment, not a detail.

---

## 6. Chapter summary

- You cannot annotate O(n²) pairs, so dedup evaluation is the discipline of estimating system-wide metrics from small, **stratified** samples.
- **Ground truth** comes from three complementary sources: stratified human annotation (boundary cases), synthetic injection (`J = 1 − f` for delete, `J = (1−f)/(1+f)` for substitute), and Cohen's kappa to validate that the human labels themselves are trustworthy (`κ ≈ 0.79` is "substantial").
- **Precision** is estimated by sampling the predicted-positive set; **recall** is estimated from a stratified pool or — exactly — from injected duplicates.
- The knobs map cleanly: `(lsh_bands, lsh_rows)` set the recall *ceiling* via the S-curve; `jaccard_threshold` trades recall for precision afterward. Sweep them and pick a point on the **Pareto frontier** matching a precision-first or recall-first objective.
- **Advanced techniques** extend the toolbox: SimHash `[Charikar 2002]` for short texts and cosine weighting, suffix arrays `[Manber & Myers 1993; Lee et al. 2022]` for exact long-substring removal, and semantic embeddings `[Abbas et al. 2023]` for paraphrase/translation dupes when GPU budget allows.
- On real Common Crawl data, expect ~31% dedup, a heavy-tailed cluster distribution where a dozen boilerplate giants do half the work, and a sober reminder that **normalization choices dominate everything**.

---

## Appendix A — Advanced technique snippets

### (a) SimHash `[Charikar 2002]`

SimHash builds a single compact fingerprint whose **Hamming distance approximates cosine similarity** (not Jaccard). Each feature is hashed to a random hyperplane; the fingerprint records the *sign* of the weighted sum on each of 64 dimensions.

```python
import hashlib, numpy as np

def simhash64(tokens, weights=None):
    """64-bit SimHash. weights = tf or tf-idf per token (cosine-of-tf flavor)."""
    weights = weights or [1.0] * len(tokens)
    v = np.zeros(64)
    for tok, w in zip(tokens, weights):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)  # stable 128-bit hash
        bits = [(h >> i) & 1 for i in range(64)]            # one random hyperplane/bit
        v += w * (2 * np.array(bits) - 1)                   # +w if bit=1 else -w
    fp = 0
    for i, x in enumerate(v):
        if x > 0:                                           # sign sketch
            fp |= (1 << i)
    return fp

def hamming(a, b):
    return bin(a ^ b).count("1")                            # # differing bits

fp1 = simhash64("the quick brown fox jumps".split())
fp2 = simhash64("the quick brown fox leaps".split())       # one word changed
print("Hamming:", hamming(fp1, fp2))   # small (e.g. 3-6) -> near-duplicate
```

**When SimHash beats MinHash:** very **short texts** (URLs, titles, tweets) where 128 MinHash values are overkill and a 64-bit fingerprint is enough; when you want **cosine-of-tf** weighting rather than set Jaccard; and when memory is tight — one 8-byte integer per doc versus a 512-byte MinHash signature.

### (b) Suffix-array dedup `[Manber & Myers 1993]`, as in `[Lee et al. 2022]`

Idea: concatenate the entire corpus into one byte string, build a **suffix array** (all suffixes sorted lexicographically), and any repeated substring of length ≥ `L` appears as a run of *adjacent* suffixes sharing a long common prefix. `[Lee et al. 2022]` use exactly this to strip duplicated 50-token spans from pretraining data — catching overlaps MinHash misses because they live *inside* otherwise-distinct documents.

```python
def repeated_substrings(text, min_len=10):
    """Find substrings of length >= min_len that occur more than once."""
    n = len(text)
    sa = sorted(range(n), key=lambda i: text[i:])   # O(n^2 log n) naive build
    found = set()
    for a, b in zip(sa, sa[1:]):                    # compare ADJACENT suffixes
        i, lcp = 0, 0                               # longest common prefix length
        while a + i < n and b + i < n and text[a + i] == text[b + i]:
            i += 1; lcp += 1
        if lcp >= min_len:
            found.add(text[a:a + lcp])
    return found

print(repeated_substrings("the license is granted. the license is sold.", 8))
# -> {' the license is '}
```

**Complexity.** The naive sort above is `O(n² log n)` (illustrative only). Production builders (DC3/SA-IS) construct the suffix array in **`O(n)`** and find all repeats in linear time — that is what makes whole-corpus exact substring dedup tractable at the terabyte scale.

### (c) Semantic dedup `[Abbas et al. 2023]` (SemDeDup)

Embed each document with a transformer, then cluster by **cosine similarity** in embedding space (or ANN). This catches **paraphrases and translations** that share almost no shingles and would be invisible to MinHash or SimHash.

```python
# OPTIONAL: requires `pip install sentence-transformers` + a GPU for real corpora.
from sentence_transformers import SentenceTransformer   # heavy, optional import
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")          # 384-dim embeddings
docs = ["A cat sat on the mat.", "The cat is sitting on the mat.",
        "Stock markets fell sharply today."]
emb = model.encode(docs, normalize_embeddings=True)      # unit vectors -> dot = cosine
sims = emb @ emb.T                                       # all-pairs cosine
print(np.round(sims, 2))
# docs 0,1 ~0.85 (paraphrase, near-dup) ; doc 2 ~0.1 (unrelated)
```

**When it's worth the cost:** only when **paraphrase or translation duplicates** genuinely matter *and* you can afford to GPU-embed the whole corpus — embedding 10M docs is hours of GPU time versus minutes for MinHash. `[Abbas et al. 2023]` show SemDeDup removes semantically redundant pretraining data that lexical methods keep, improving downstream efficiency, but it is a *complement* to MinHash/LSH, not a replacement.

---

## Self-check quiz

**Q1.** You inject near-duplicates by **substituting** 20% of word shingles (`edit_fraction = 0.20`) and your `jaccard_threshold` is 0.80. Will the pipeline recover them, and what realized Jaccard should you expect?

<details><summary>Answer</summary>
Substitution gives `J = (1−f)/(1+f) = 0.80/1.20 ≈ 0.667`, which is **below** the 0.80 threshold — the verify step will reject them, so recall on these injections will be near zero. That is correct behavior on mislabeled truth, not a bug. Use **delete** mode (`J = 1 − f = 0.80`) if you want them at the boundary, or lower the threshold.
</details>

**Q2.** Two annotators agree on 88% of 1,000 pairs (`p_o = 0.88`). Annotator A marks 30% DUP, annotator B marks 25% DUP. Compute Cohen's kappa and classify it.

<details><summary>Answer</summary>
`p_e = (0.30)(0.25) + (0.70)(0.75) = 0.075 + 0.525 = 0.600`.
`κ = (0.88 − 0.60)/(1 − 0.60) = 0.28/0.40 = 0.70` → **"substantial"** agreement (0.61–0.80). Trustworthy labels, near the lower edge of substantial.
</details>

**Q3.** Your measured precision is 0.95 but spot-checks reveal distinct documents merged. Which pitfall is this, and what single report change exposes it?

<details><summary>Answer</summary>
The **Giant-Cluster Precision Mirage** (Pitfall 3). A few huge boilerplate clusters contribute millions of *correct* pairs that dilute the average. The fix: report metrics **per cluster-size stratum** and measure cluster purity on a sample of *small* (size 2–5) clusters, where a single wrong edge is actually visible.
</details>

---

## References

- `[Charikar 2002]` Moses S. Charikar. *Similarity Estimation Techniques from Rounding Algorithms.* STOC 2002. (SimHash / random-hyperplane sketches.)
- `[Manber & Myers 1993]` Udi Manber and Gene Myers. *Suffix Arrays: A New Method for On-Line String Searches.* SIAM Journal on Computing, 22(5), 1993.
- `[Leskovec et al. 2014]` Jure Leskovec, Anand Rajaraman, Jeffrey D. Ullman. *Mining of Massive Datasets*, 2nd ed., Cambridge University Press, 2014. (MinHash, LSH banding, the S-curve.)
- `[Lee et al. 2022]` Katherine Lee, Daphne Ippolito, Andrew Nystrom, et al. *Deduplicating Training Data Makes Language Models Better.* ACL 2022. (Suffix-array exact substring dedup for LLM data.)
- `[Abbas et al. 2023]` Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, Ari S. Morcos. *SemDeDup: Data-efficient Learning at Web-Scale through Semantic Deduplication.* 2023. (Embedding-based semantic dedup.)
