# Chapter 1: Why Data Curation Matters

> Module: *Build Your Own Data Curation Pipeline (MinHash / Deduplication)*
> Audience: mid-level software engineers with **no prior NLP background**.

Before you write a single line of MinHash code, you need to internalize *why* deduplication is one of the highest-leverage things you can do to a pretraining corpus. This chapter builds that intuition from first principles, grounds it in published numbers, and ends with a runnable script that measures duplication in a real corpus you generate yourself.

---

## 1. Learning objectives

By the end of this chapter you will be able to:

- **Explain** the causal chain from *dataset quality* to *model quality*, using empirical results from scaling-laws research to argue why deduplication both improves downstream accuracy and *saves* training compute.
- **Define** the three harms caused by duplicate data — verbatim memorization, bias amplification, and benchmark contamination — and identify which one dominates in a given scenario.
- **Classify** any pair of documents into the duplicate taxonomy (exact / near-exact / semantic) and decide whether that pair is worth removing.
- **Quantify** duplication: cite approximate published dedup ratios for Common Crawl, C4, and MassiveWeb, and reason about why web data is so redundant.
- **Measure** exact and near-duplicate rates in a small corpus yourself using a cheap hashing + MinHash-style sketch, the same primitive you will productionize in `dedup_pipeline.minhash.minhash.MinHasher`.

---

## 2. Concept explanation

### 2.1 The core claim: the model is a compression of its data

A large language model has no knowledge other than what is in its training set. During pretraining it does maximum-likelihood next-token prediction over the corpus, so the corpus's statistical structure *is* the model's prior. If a sentence appears 1,000 times, the model sees gradient updates pushing toward that sentence 1,000 times more strongly than a sentence that appears once. **Duplication is therefore an unintentional, uncontrolled reweighting of your training distribution.** Nobody chose to upweight "Click here to accept cookies" by 50,000×, but a raw web crawl does exactly that.

### 2.2 Dataset quality vs. model quality, with real numbers

The classical scaling laws of [Kaplan et al. 2020] and the compute-optimal refinement of [Hoffmann et al. 2022] (Chinchilla) describe loss as a smooth power law in three quantities: model parameters `N`, dataset size `D`, and compute `C`. The Chinchilla finding is concrete: for a fixed compute budget, the optimal allocation scales `N` and `D` *in roughly equal proportion*. Their headline example: a **70B**-parameter model trained on **1.4 trillion tokens** (Chinchilla) outperformed the **280B**-parameter Gopher trained on ~300B tokens — a 4× smaller model beating a model trained with the same compute, simply by training on more, better-balanced data.

Here is the subtle part those laws assume: `D` is the count of *effective, distinct* tokens. A duplicate token is not a new observation — it carries almost no new information — yet it still consumes a forward/backward pass. So duplication silently inflates your nominal `D` while leaving your *effective* `D` unchanged. Worked example with round numbers:

- You crawl **1,000,000,000** (1B) documents averaging **400 tokens** each → **400B nominal tokens**.
- Suppose **30%** of documents are near-duplicates of something else (a realistic web figure, see §2.5).
- After deduplication you keep **700,000,000** documents → **280B effective tokens**.
- You just discarded **120B tokens** of training *that would have cost real GPU-hours and delivered near-zero marginal learning.* At a throughput of, say, **2,000 tokens/sec/GPU**, those 120B tokens are **120e9 / 2000 = 60,000,000 GPU-seconds ≈ 16,667 GPU-hours** of pure waste per epoch.

The empirical payoff is documented directly. [Lee et al. 2022] ("Deduplicating Training Data Makes Language Models Better") trained models on deduplicated vs. raw corpora and found that **deduplicated training reaches the same or better validation perplexity with up to ~10× fewer parameter updates on the duplicated content**, and improves held-out perplexity. The two effects compound: you train faster *and* the resulting model is better.

### 2.3 Harm #1 — Memorization and verbatim regurgitation

Neural networks memorize. [Carlini et al. 2021] demonstrated *training-data extraction*: by prompting GPT-2 they recovered verbatim sequences — names, phone numbers, UUIDs, code — that appeared in the training data. The crucial follow-up, [Carlini et al. 2022], established a clean quantitative law: **memorization grows log-linearly with the number of times a sequence is duplicated.** A string seen once is rarely emitted verbatim; a string seen hundreds of times is emitted with high probability. Concretely, [Lee et al. 2022] showed that a model trained on deduplicated data regurgitated memorized training text **roughly 10× less often** than the same model trained on the raw corpus. Memorization is mostly a *duplication* problem, and deduplication is the cheapest mitigation — cheaper than differential privacy, which degrades accuracy.

### 2.4 Harm #2 — Bias amplification, and Harm #3 — Benchmark contamination

**Bias amplification.** If a single opinionated template ("As an SEO expert, the best product is…") is spammed across 100,000 pages, the model treats that framing as a strong prior. Duplicates turn the loudest, most-copied voices on the web into the model's default voice. Deduplication flattens this skew back toward one-document-one-vote.

**Benchmark / test-set contamination.** This is the one that quietly invalidates your evaluation. If a copy of a benchmark question (say, a GSM8K math problem or an MMLU item) leaked into the training crawl, the model may have *memorized the answer* rather than learned to solve it. Your reported accuracy is then inflated and meaningless. [Lee et al. 2022] and [Raffel et al. 2020] both flag train/test overlap as a first-class hazard; the standard defense is to deduplicate the training set *against the evaluation sets* using the same near-duplicate machinery, removing any training document that closely matches a test item.

### 2.5 Taxonomy of duplicates (definitions + concrete examples)

| Type | Definition | Example | When it matters | When you can ignore it |
|---|---|---|---|---|
| **(a) Exact** | Byte-for-byte identical after reading raw bytes | Two crawls of `https://site/page` returning the identical HTML body | Always remove — pure waste, trivially detected with one hash | Almost never ignore; cost of removal is near-zero |
| **(b) Near-exact** | Differ only by minor edits, whitespace, encoding, boilerplate, headers/footers, ad injection, timestamps | Same article on two mirror sites, one with a cookie banner and a different ad sidebar | The dominant web case — this is what MinHash targets | If you *intentionally* keep boilerplate (e.g., training a layout model) |
| **(c) Semantic** | Same information, different surface words | "The cat sat on the mat." vs. "On the mat, a cat was sitting." Or a paraphrased news summary | Diversity-sensitive tasks; eval contamination via paraphrase | Often ignored in pretraining — paraphrases add genuine linguistic variety and removing them needs expensive embeddings |

The practical sweet spot for petabyte-scale pretraining is **(b) near-exact** detection. (a) is too narrow (catches almost nothing on a messy crawl); (c) is too expensive and too aggressive (you may delete useful paraphrase diversity, and it requires embedding models). MinHash + LSH lives precisely at level (b): it catches exact and near-exact duplicates cheaply, at scale, with a tunable similarity threshold.

### 2.6 Real-world case study: how bad is it, really?

These are **approximate published figures** — exact numbers depend on the dedup method and threshold, so treat them as orders of magnitude:

- **Common Crawl (raw web):** routinely reported at **30%+ duplicated content**; for some crawls and document-level analyses, a large fraction (in places ~40–50%) of content is redundant once near-duplicates are counted. The web is full of mirrors, syndication, scrapers, and templated pages.
- **C4** (the Colossal Clean Crawled Corpus of [Raffel et al. 2020], a *cleaned* slice of Common Crawl): even after aggressive heuristic filtering, [Lee et al. 2022] found **substantial near-duplication remaining** — long spans of text repeated across many documents, and on the order of a few percent (~3%+) of documents removable as near-duplicates by their NearDup method, with vastly more *substring*-level repetition detectable by suffix-array ExactDup.
- **MassiveWeb / Gopher** ([Rae et al. 2021]): DeepMind's pipeline applied explicit document-level near-deduplication as a *named, mandatory* stage, removing a meaningful fraction of documents; they report deduplication as essential to corpus quality. The takeaway is not the precise percentage but that *every* serious lab treats dedup as non-optional.

The pattern across all three: **raw crawls are tens-of-percent duplicated; even "clean" datasets retain single-digit-percent document duplication and far more span-level repetition.** That is why this entire course exists.

---

## 3. Annotated code walkthrough (and hands-on exercise)

The script below builds a synthetic 300-document corpus with *known* injected duplicates, then measures **exact** and **near-duplicate** rates two ways: (1) exact via full-document hashing, (2) near-duplicate via a cheap MinHash-style sketch over character shingles. This is a stripped-down preview of `dedup_pipeline.text_processing.shingler` (shingling) and `dedup_pipeline.minhash.minhash.MinHasher` (sketching), which the full `dedup_pipeline.pipeline.pipeline.DedupPipeline` orchestrates at scale.

It runs on the **standard library only** (`hashlib`, `random`, `re`). No third-party packages required.

```python
"""Chapter 1 hands-on: measure exact and near-duplicate rates in a tiny corpus.

Run with:  python chapter_01_dedup_demo.py
Standard library only.
"""
import hashlib
import random
import re

random.seed(42)  # Determinism: every reader gets identical numbers. Real pipelines
                 # pin a seed too (PipelineConfig.random_seed) so runs are reproducible.

# ---------------------------------------------------------------------------
# 1. Build a corpus of 300 documents with KNOWN duplicates injected.
#    We know ground truth, so we can later check that our measurement is right.
# ---------------------------------------------------------------------------
BASE_SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning models compress their training data",
    "deduplication improves downstream model accuracy and saves compute",
    "common crawl contains a large fraction of duplicate web pages",
    "minhash estimates jaccard similarity using a single minimum value",
]

corpus = []
# 250 "unique" docs: each is a base sentence padded with random filler words so
# that no two are byte-identical, but some pairs remain highly similar.
for i in range(250):
    base = random.choice(BASE_SENTENCES)
    filler = " ".join(f"w{random.randint(0, 9999)}" for _ in range(random.randint(3, 8)))
    corpus.append(f"{base} {filler}")

# Inject 30 EXACT duplicates: clone existing docs byte-for-byte.
for _ in range(30):
    corpus.append(random.choice(corpus[:250]))

# Inject 20 NEAR-exact duplicates: clone a doc but mutate whitespace/casing/punct.
for _ in range(20):
    original = random.choice(corpus[:250])
    near = original.upper()                       # casing change
    near = re.sub(r"\s+", "   ", near)            # collapse/expand whitespace
    near = near + " ."                            # trailing punctuation/boilerplate
    corpus.append(near)

random.shuffle(corpus)  # Mix them in; a real crawl gives no hints about which is which.
N = len(corpus)         # 250 + 30 + 20 = 300 documents.

# ---------------------------------------------------------------------------
# 2. EXACT duplicate rate: hash the raw bytes of each document.
#    Two docs collide IFF they are byte-for-byte identical (ignoring astronomically
#    rare SHA-1 collisions). We group by hash and count docs beyond the first
#    in each group as removable duplicates.
# ---------------------------------------------------------------------------
def exact_key(doc: str) -> str:
    # Encode to bytes first: hashing is defined over bytes, not Python str objects.
    return hashlib.sha1(doc.encode("utf-8")).hexdigest()

seen, exact_dups = set(), 0
for doc in corpus:
    k = exact_key(doc)
    if k in seen:
        exact_dups += 1        # a copy of something we already counted -> removable
    else:
        seen.add(k)
print(f"Exact duplicates removable: {exact_dups} / {N} = {exact_dups / N:.1%}")

# ---------------------------------------------------------------------------
# 3. NEAR-duplicate rate via a cheap MinHash sketch.
#    Step 3a: normalize so trivial differences (case, whitespace) don't fool us.
#    This mirrors dedup_pipeline.text_processing normalization.
# ---------------------------------------------------------------------------
def normalize(doc: str) -> str:
    doc = doc.lower()                     # casing no longer matters
    doc = re.sub(r"[^\w\s]", "", doc)     # drop punctuation (boilerplate dots, etc.)
    doc = re.sub(r"\s+", " ", doc).strip()# collapse all whitespace runs to one space
    return doc

# Step 3b: shingling. A k-shingle is a length-k sliding window over characters.
#   "abcd" with k=3 -> {"abc", "bcd"}. Shingle SETS overlap heavily for similar docs.
def shingles(doc: str, k: int = 5) -> set[str]:
    s = normalize(doc)
    if len(s) < k:
        return {s}                        # very short doc: treat whole string as 1 shingle
    return {s[i:i + k] for i in range(len(s) - k + 1)}

# Step 3c: a MinHash sketch. For each of NUM_HASHES independent hash "salts",
#   we take the MINIMUM hash value over all shingles. Broder's key theorem:
#   P(min hash of A == min hash of B) == Jaccard(A, B). So the fraction of the
#   NUM_HASHES positions where two sketches agree ESTIMATES their Jaccard similarity.
NUM_HASHES = 64
def minhash_sketch(doc: str) -> tuple[int, ...]:
    shs = shingles(doc)
    sketch = []
    for salt in range(NUM_HASHES):
        best = None
        for sh in shs:
            # Salt the shingle so each of the 64 "hash functions" orders shingles
            # differently; md5 here stands in for a fast universal hash family.
            h = int(hashlib.md5(f"{salt}:{sh}".encode("utf-8")).hexdigest(), 16)
            if best is None or h < best:
                best = h
        sketch.append(best if best is not None else 0)
    return tuple(sketch)

def estimate_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    # Fraction of sketch positions that agree == unbiased estimate of Jaccard.
    agree = sum(1 for x, y in zip(a, b) if x == y)
    return agree / len(a)

# Step 3d: count near-duplicates. NOTE: this O(N^2) all-pairs loop is fine for
#   N=300 but is exactly what LSH (Chapter 4) eliminates at scale.
sketches = [minhash_sketch(doc) for doc in corpus]
THRESHOLD = 0.8                # call a pair "near-duplicate" if estimated Jaccard >= 0.8
near_dup_docs = set()
for i in range(N):
    if i in near_dup_docs:
        continue               # i already counted as someone's duplicate; skip it
    for j in range(i + 1, N):
        if j in near_dup_docs:
            continue           # don't double-count j
        if estimate_jaccard(sketches[i], sketches[j]) >= THRESHOLD:
            near_dup_docs.add(j)   # j is a near-duplicate of an earlier doc i
print(f"Near-duplicates (incl. exact) removable: "
      f"{len(near_dup_docs)} / {N} = {len(near_dup_docs) / N:.1%}")
```

**What you should observe.** Exact hashing recovers about **30 / 300 = ~10%** removable exact duplicates. The MinHash sketch recovers those *plus* the 20 normalized near-duplicates *plus* some base-sentence pairs that happen to be ≥0.8 similar — landing around **15–20% or higher**. The gap between the two numbers is precisely the value MinHash adds over naive hashing: it catches the near-exact bucket (taxonomy type b) that exact hashing is blind to.

---

## 4. Common pitfalls

### Pitfall 1 — "Exact hashing is enough" (the under-detection trap)
**Diagnosis.** You hash full documents with SHA-1, report a tiny duplicate rate (e.g., 2%), and conclude your data is clean. But your downstream perplexity barely improves and the model still regurgitates training text. The cause: a single changed byte — an updated timestamp, a different ad, a cookie banner — makes two otherwise-identical pages hash differently, so exact hashing misses the entire near-exact bucket, which is where most web redundancy lives ([Lee et al. 2022]).
**Fix.** Move to similarity-based detection: normalize first (case, whitespace, punctuation, HTML), shingle, and use MinHash with a Jaccard threshold (~0.8). This is the whole point of `dedup_pipeline.minhash.minhash.MinHasher`.

### Pitfall 2 — Over-aggressive threshold collapses your dataset (the over-detection trap)
**Diagnosis.** You set the Jaccard threshold too low (e.g., 0.3) or use very long shingles incorrectly, and suddenly 60% of your corpus is flagged as duplicate. Distinct documents that merely share common phrases or boilerplate get clustered together and deleted, destroying genuine diversity and *hurting* downstream accuracy.
**Fix.** Pick the threshold deliberately (Chapter 4 derives the LSH S-curve for a target threshold). Validate on a small labeled sample: inject duplicates at known similarities (e.g., 0.85) and confirm your pipeline flags those but spares dissimilar pairs. Treat the threshold as a precision/recall dial, not a constant.

### Pitfall 3 — Forgetting to deduplicate train against test (the contamination trap)
**Diagnosis.** Your model scores suspiciously high on a public benchmark but underperforms in production. You deduplicated the training set internally but never checked it against the evaluation sets, so memorized test items leaked through and inflated your scores ([Lee et al. 2022], [Raffel et al. 2020]).
**Fix.** Run the *same* near-duplicate machinery between the training corpus and every held-out/benchmark set, and drop any training document that closely matches a test item. Do this *before* reporting any benchmark number.

### Pitfall 4 (bonus) — Normalization mismatch between stages
**Diagnosis.** Your dedup rate is unstable across runs, or two clearly identical docs aren't matched. Different stages normalized text differently (one lowercased, one didn't), so their shingle sets diverge and Jaccard is artificially low.
**Fix.** Centralize normalization in one place (here, the `normalize()` function; in production, `dedup_pipeline.text_processing`) and apply it identically before shingling *everywhere*.

---

## 5. Chapter summary

A pretrained model is a compression of its training distribution, so **uncontrolled duplication is uncontrolled reweighting** of that distribution. Scaling laws ([Kaplan et al. 2020], [Hoffmann et al. 2022]) tell us that only *effective, distinct* tokens drive learning, so duplicated tokens waste compute while adding no information — [Lee et al. 2022] showed deduplication reaches equal-or-better quality with far fewer updates and cuts verbatim memorization roughly tenfold ([Carlini et al. 2022]). Duplication causes three concrete harms: **memorization/regurgitation**, **bias amplification**, and **benchmark contamination**. Duplicates come in three flavors — **exact** (byte-identical), **near-exact** (minor edits/boilerplate), and **semantic** (paraphrase) — and the high-value, tractable target for large-scale pretraining is the **near-exact** bucket, which is exactly what MinHash + LSH addresses. Published figures confirm the stakes: raw crawls like Common Crawl are **30%+** duplicated, and even cleaned corpora like C4 ([Raffel et al. 2020]) and MassiveWeb ([Rae et al. 2021]) retain meaningful document- and span-level redundancy. You built and ran a standard-library script that measured both exact (~10%) and near-duplicate (~15–20%+) rates, previewing the shingling and sketching primitives you will engineer for scale in the chapters ahead.

---

## 6. Self-check quiz

**Q1.** A teammate runs full-document SHA-256 hashing over a 10M-document Common Crawl shard, finds a 1.5% duplicate rate, and declares the data "basically clean." Why is this conclusion almost certainly wrong, and what is the single most likely category of duplicates they are missing?

> **A1.** Exact hashing only catches **byte-for-byte identical** documents (taxonomy type *a*). A single differing byte — a timestamp, ad, cookie banner, or whitespace change — makes near-identical pages hash differently. They are missing the **near-exact** bucket (type *b*), which is where the majority of web redundancy lives; published analyses put real crawl duplication at **30%+** once near-duplicates are counted ([Lee et al. 2022]). The fix is similarity-based detection (normalize → shingle → MinHash).

**Q2.** Using round numbers, explain why training on a 30%-duplicated 400B-token corpus is wasteful. Roughly how many "effective" distinct tokens does the model actually learn from, and what is the consequence in light of Chinchilla-style scaling?

> **A2.** If 30% of content is duplicated, only ~70% of the 400B tokens are distinct → ~**280B effective tokens**; the other ~120B tokens consume forward/backward passes while adding almost no new information. Because compute-optimal scaling ([Hoffmann et al. 2022]) treats `D` as *distinct* data, those 120B duplicate tokens are wasted compute that neither improves loss nor adds knowledge — and they actively *increase* memorization/regurgitation ([Carlini et al. 2022]).

**Q3.** You set the MinHash Jaccard threshold to 0.3 and your pipeline flags 65% of documents as duplicates. Is this likely correct? Name the failure mode and state the fix.

> **A3.** No — this is the **over-detection trap** (Pitfall 2). A threshold of 0.3 is far too permissive: documents that merely share common phrases or boilerplate exceed it and get wrongly clustered and deleted, destroying genuine diversity. The fix is to choose the threshold deliberately (a value around **0.8** for near-exact detection, tuned via the LSH S-curve in Chapter 4) and validate it against duplicates injected at known similarities.

---

## References

- **[Kaplan et al. 2020]** Kaplan, J., McCandlish, S., Henighan, T., et al. *Scaling Laws for Neural Language Models.*
- **[Hoffmann et al. 2022]** Hoffmann, J., Borgeaud, S., Mensch, A., et al. *Training Compute-Optimal Large Language Models* (Chinchilla).
- **[Lee et al. 2022]** Lee, K., Ippolito, D., Nystrom, A., et al. *Deduplicating Training Data Makes Language Models Better.*
- **[Carlini et al. 2021]** Carlini, N., Tramèr, F., Wallace, E., et al. *Extracting Training Data from Large Language Models.*
- **[Carlini et al. 2022]** Carlini, N., Ippolito, D., Jagielski, M., et al. *Quantifying Memorization Across Neural Language Models.*
- **[Raffel et al. 2020]** Raffel, C., Shazeer, N., Roberts, A., et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* (T5 / C4).
- **[Rae et al. 2021]** Rae, J. W., Borgeaud, S., Cai, T., et al. *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* (MassiveWeb).
