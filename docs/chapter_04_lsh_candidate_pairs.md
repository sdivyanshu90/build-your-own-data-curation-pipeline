# Chapter 4: Locality-Sensitive Hashing (LSH) for Candidate Pair Generation

> **A note on the numbers in this chapter.** Every probability, threshold, and
> arithmetic result below was computed by hand and is shown step-by-step so you
> can reproduce it with a calculator. Where a popular rule-of-thumb disagrees with
> the exact arithmetic, we trust the arithmetic and say so explicitly.

---

## 1. Learning objectives

By the end of this chapter you will be able to:

- Explain **why MinHash signatures, on their own, do not solve the quadratic blowup**, and put a wall-clock number on the problem for a 1M-document corpus.
- Derive from first principles the **LSH banding probability** `P(candidate) = 1 − (1 − s^r)^b` and explain every term.
- Read and tune the **S-curve**: locate its threshold, and trade recall against precision by adjusting the number of bands `b` and rows per band `r`.
- **Choose `(b, r)` for a target similarity threshold** under a fixed signature length `n`, and verify the operating point numerically.
- Build an **inverted bucket index**, deduplicate emitted pairs with a **Bloom filter** [Bloom 1970], and **partition the work across nodes** by band so full signatures never shuffle.

---

## 2. Concept explanation

### 2.1 MinHash compressed the *document*, not the *comparison*

In Chapter 3 we replaced each document's shingle set with a fixed-length MinHash
**signature** of `n` integers. The signature is tiny — a few hundred bytes — and
the fraction of matching positions between two signatures is an unbiased estimate
of the Jaccard similarity `s` of the original sets [Broder 1997].

That solved *storage*. It did **not** solve *comparison*. To find all near-duplicate
pairs you still, naively, compare every signature to every other signature. For
`n_docs` documents that is

```
n_docs · (n_docs − 1) / 2   pairs.
```

This is the same `O(n_docs²)` wall we hit with exact Jaccard — only the per-pair
cost shrank.

**Wall-clock problem, 1M documents.** With `n_docs = 1,000,000`:

```
pairs = 1,000,000 · 999,999 / 2 = 499,999,500,000  ≈ 5.0 × 10¹¹ pairs.
```

Suppose comparing two 128-integer signatures takes a (generous) 100 nanoseconds
on one core. Then:

```
5.0 × 10¹¹ pairs × 100 ns = 5.0 × 10¹³ ns = 5.0 × 10⁴ seconds ≈ 13.9 hours.
```

Roughly **14 core-hours for one million documents** — and the cost grows with the
*square* of the corpus. At 10M documents it is ~1,390 core-hours; at 100M it is
~139,000 core-hours. Pretraining corpora have *billions* of documents. Brute force
is a non-starter. We need a way to look only at pairs that are *plausibly* similar.

### 2.2 The banding trick: only compare documents that already collide somewhere

Locality-Sensitive Hashing [Indyk & Motwani 1998] is the family of techniques that
make near-neighbor search sub-quadratic. The variant that pairs naturally with
MinHash is **banding** [Leskovec et al. 2014].

Take the length-`n` signature and chop it into **`b` bands of `r` rows each**, so
that

```
n = b · r.
```

For each band (a slice of `r` consecutive signature values), hash the whole
`r`-tuple to a bucket id. Two documents are declared a **candidate pair** if they
land in the *same bucket in at least one band*.

The intuition is a logical AND nested inside a logical OR:

- **AND inside a band:** to collide in one band, *all `r` values* in that band must
  match. One mismatch in the band and the `r`-tuples differ, so they hash to
  different buckets (collisions aside — see pitfalls).
- **OR across bands:** you get `b` independent chances. Colliding in *any one* of
  the `b` bands is enough.

So a band is a stringent test ("agree on all `r` rows"), but we run the test `b`
times and accept on the first success. High-similarity documents pass at least one
band almost surely; low-similarity documents fail all of them almost surely. That
separation is exactly what we want.

### 2.3 Full derivation of `P(candidate)`

Let `s` be the true Jaccard similarity of two documents. Recall the MinHash
guarantee: for any single signature row,

```
P(the two signatures agree on that row) = s.                       (1)
```

MinHash rows are constructed from independent hash functions, so rows are
independent. Now walk up the band structure:

**Step 1 — agree on all `r` rows of one band.** Independence over the `r` rows:

```
P(band matches) = s · s · … · s  (r times) = s^r.                  (2)
```

**Step 2 — that band differs.** Complement of (2):

```
P(band differs) = 1 − s^r.                                         (3)
```

**Step 3 — all `b` bands differ.** Bands use disjoint rows, hence independent:

```
P(all b bands differ) = (1 − s^r)^b.                               (4)
```

**Step 4 — at least one band matches (= becomes a candidate).** Complement of (4):

```
P(candidate) = 1 − (1 − s^r)^b.                                    (5)
```

Equation (5) is the heart of LSH. It says the probability of *catching* a pair is a
function only of `s`, `b`, and `r`.

### 2.4 The S-curve

Plot equation (5) against `s ∈ [0, 1]` and you get a sigmoid ("S") shape: nearly
flat and low for small `s`, a steep ramp in the middle, then nearly flat and high
near `s = 1`. The steep region is the **threshold** — the similarity above which a
pair is very likely caught and below which it is very likely dropped.

A useful (approximate) location of the steep region is

```
t ≈ (1 / b)^(1 / r).                                               (6)
```

**Worked threshold for `b = 16`, `r = 8`.** Because `16 = 2⁴`,

```
t = (1/16)^(1/8) = 16^(−1/8) = (2⁴)^(−1/8) = 2^(−1/2) = 1/√2 ≈ 0.7071.
```

> ⚠️ Some course notes quote this same `(b=16, r=8)` threshold as ≈ 0.6866. That
> value does not satisfy equation (6): `(1/16)^(1/8)` is exactly `2^(−1/2) = 0.7071`,
> not 0.6866. We use **0.7071**. Always recompute (6) for your own `(b, r)` rather
> than trusting a memorized constant.

**The probability table for `(b = 16, r = 8)`.** Using (5), `P = 1 − (1 − s⁸)¹⁶`.
Each row shows the intermediate `s⁸` so you can audit it:

| `s` | `s⁸` | `1 − s⁸` | `(1 − s⁸)¹⁶` | **`P(candidate)`** |
|----:|-----:|---------:|-------------:|-------------------:|
| 0.5 | 0.003906 | 0.996094 | 0.93930 | **0.0607** |
| 0.6 | 0.016796 | 0.983204 | 0.76265 | **0.2374** |
| 0.7 | 0.057648 | 0.942352 | 0.38688 | **0.6131** |
| 0.8 | 0.167772 | 0.832228 | 0.05287 | **0.9471** |
| 0.9 | 0.430467 | 0.569533 | 0.00012 | **0.9999** |

Worked check of the `s = 0.8` cell: `0.8⁸ = 0.16777216`, so `1 − 0.8⁸ = 0.83222784`.
Then `0.83222784¹⁶`: `ln(0.83222784) = −0.183683`, times 16 is `−2.93893`,
`e^(−2.93893) = 0.052874`. Finally `P = 1 − 0.052874 = 0.94713`.

> ⚠️ A second common misquote claims `s = 0.8 → P ≈ 0.9997` for `(b=16, r=8)`. The
> exact value is **0.9471**. Reaching 0.9997 at `s = 0.8` would require about
> `b ≈ 44` bands at `r = 8` (`44 · ln(0.83223) ≈ −8.1`, and `1 − e^(−8.1) ≈ 0.9997`).
> The lesson: the operating point is *very* sensitive to `b`; never eyeball it.

**ASCII S-curve, `(b = 16, r = 8)`** (each `#` ≈ 2% probability):

```
P(candidate)
1.0 |                               #################  s=0.90 (0.9999)
    |                          ##### #################  s=0.85 (0.9936)
0.8 |                     #####                         s=0.80 (0.9471)
    |                #####                              s=0.75 (0.8151)
0.6 |            ####                                   s=0.70 (0.6131)
    |          ##                                       s=0.65 (0.4044)
0.4 |        ##                                         s=0.60 (0.2374)
    |      ##                                           s=0.55 (...)
0.2 |    ##                                             s=0.50 (0.0607)
    | ###                                               s≤0.45 (~0)
0.0 +--------------------------------------------------
     0.0   0.2    0.4    0.5    0.6   0.7   0.8   0.9  1.0   →  s
```

**Precision / recall trade-off.** Two knobs move the curve:

- **More bands `b`** (more OR chances) shifts the whole curve *left* and *up*: more
  pairs clear the bar, so **recall rises but precision falls** — you generate more
  candidates, including more false positives that the exact-similarity verification
  stage must later reject.
- **More rows per band `r`** (stricter AND) makes the curve *steeper* and pushes the
  threshold *right*: the transition from "rejected" to "accepted" sharpens, which
  tightens precision but can drop recall just below the threshold.

You tune `b` to set *where* the curve sits and `r` to set *how sharp* it is, subject
to the budget `n = b · r`.

### 2.5 Choosing `(b, r)` for a target threshold (`t = 0.8`, fixed `n = 128`)

We want pairs with `s ≥ 0.8` caught reliably, and we have a 128-integer signature
budget, so `b · r = 128`. Candidate factorizations and their thresholds via (6):

| `b` | `r` | `t ≈ (1/b)^(1/r)` | `P(s=0.7)` | `P(s=0.8)` | `P(s=0.9)` |
|----:|----:|------------------:|-----------:|-----------:|-----------:|
| 8  | 16 | `2^(−3/16)`  ≈ 0.8787 | 0.0064 | 0.2042 | 0.8723 |
| 16 | 8  | `2^(−1/2)`   ≈ 0.7071 | 0.6131 | **0.9471** | 0.9999 |
| 32 | 4  | `2^(−5/4)`   ≈ 0.4204 | 0.9978 | ~1.0000 | ~1.0000 |

- `(8, 16)` puts the threshold at 0.879 — *too strict*, only 20% recall at `s = 0.8`.
- `(32, 4)` puts the threshold at 0.420 — *too loose*, it would flag 99.8% of
  merely-0.7-similar pairs, drowning the verifier in false positives.
- **`(16, 8)`** is the right pick: threshold 0.707 (just below our 0.8 target, which
  is what we want — the curve should be *already rising* at the target), and
  `P(s = 0.8) = 0.9471`.

Verification at `s = 0.8`: from the table above, `P = 0.9471`. That is `≈ 0.95`
to two decimals. If your spec demands a *strict* `P(s=0.8) ≥ 0.95`, the honest fix
is to spend a little more signature budget: keep `r = 8`, raise to `b = 20`
(`n = 160`). Then

```
P(0.8) = 1 − 0.83222784²⁰ = 1 − e^(20 · ln 0.83222784)
       = 1 − e^(−3.67366) = 1 − 0.025398 = 0.9746  ✓ ( ≥ 0.95 ).
```

So `(b=16, r=8)` is the best you can do *inside* `n = 128`; meeting a hard 0.95 floor
costs you 32 more integers per signature. This budget-vs-recall tension is the core
engineering decision of LSH tuning.

### 2.6 Inverted bucket index + Bloom-filtered pair emission

Once each document has, per band, a bucket id, finding candidates is a *grouping*
problem, not a comparison problem:

1. Build an inverted index `bucket_id → [doc ids]`, one entry per band.
2. For each bucket that holds ≥ 2 documents, emit all pairs *within that bucket*.
3. A pair can collide in several bands, so you would emit it multiple times. Keep a
   **seen-set** — or, at scale, a **Bloom filter** [Bloom 1970] — to suppress
   duplicate emissions cheaply.

A Bloom filter is a bit array of `m` bits with `k` hash functions. Its
false-positive probability after inserting `n` elements is

```
p ≈ (1 − e^(−k·n/m))^k.                                            (7)
```

**Concrete sizing.** Say we expect `n = 10,000,000` distinct candidate pairs and
budget `10 bits per element`, i.e. `m = 100,000,000` bits = **12.5 MB**. The optimal
hash count is `k = (m/n)·ln 2 = 10 · 0.6931 ≈ 7`. Plug into (7):

```
k·n/m = 7 · 10⁷ / 10⁸ = 0.7
1 − e^(−0.7) = 1 − 0.496585 = 0.503415
p = 0.503415⁷ = e^(7 · ln 0.503415) = e^(−4.804) ≈ 0.0082  ≈ 0.82%.
```

A false positive here means "we wrongly think we already emitted this pair, so we
*drop* it." For a *pre*-filter that's a tolerable ~0.8% recall hit at 12.5 MB; if you
cannot afford to lose any pair, use an exact seen-set (a hash set) and pay the RAM,
or use the Bloom filter only to skip the *expensive* exact-Jaccard recheck while
still emitting. Choose per your precision/recall budget.

### 2.7 Distributed LSH: partition by band, not by document

The banding structure hands you a clean parallelization: **partition the work by
band**. Assign each of the `b` bands (or contiguous groups of bands) to a node. A
node receives only *its* rows of the signature matrix — an `r`-row slice per
document — does its own bucketing, builds its own local inverted index, and emits
its own local candidate pairs.

Why this is the right cut:

- **No full-signature shuffle.** Each node needs only `r` of the `n` rows. You never
  move whole signatures across the network; you move thin band slices. Network cost
  scales with `r`, not `n`.
- **Independent buckets.** A bucket lives entirely inside one band, so it lives
  entirely on one node. There is no cross-node bucket to reconcile.
- **Merge is a union.** Each pair is `(min_id, max_id)`-ordered locally, then the
  driver takes the **union** of all nodes' pair sets (dedup across bands happens at
  merge — a distributed seen-set or per-node Bloom filters reduce the volume first).

**Cross-partition pairs** are the subtle case: a single true pair may be discovered
by *different* nodes (it collided in band 3 on node A *and* band 11 on node B). That
is fine — the union step de-duplicates it. What banding guarantees is that the
*bucketing itself* never spans nodes, so the only cross-node traffic is the final,
already-compact list of candidate pairs, not the bulky signatures.

---

## 3. Annotated code walkthrough

The pipeline ships three real modules:
`dedup_pipeline.lsh.banding`, `dedup_pipeline.lsh.bucket_index.BucketIndex`, and
`dedup_pipeline.lsh.candidate_pairs`. Below are runnable, self-contained reference
implementations (numpy + stdlib) that mirror their responsibilities.

```python
"""Reference implementation of dedup_pipeline.lsh.banding."""
import math
from typing import Iterator
import numpy as np


def estimate_threshold(b: int, r: int) -> float:
    # Approximate S-curve threshold from equation (6): t ≈ (1/b)^(1/r).
    return (1.0 / b) ** (1.0 / r)


def candidate_probability(s: float, b: int, r: int) -> float:
    # Equation (5): exact probability a pair of Jaccard s becomes a candidate.
    return 1.0 - (1.0 - s ** r) ** b


def iter_bands(signature: np.ndarray, b: int, r: int) -> Iterator[tuple[int, bytes]]:
    """Yield (band_index, band_key) for one document's signature.

    band_key is the r values of the band turned into immutable bytes so it can be
    hashed into a dict. We do NOT hash to a fixed-width int bucket here: using the
    raw r-tuple bytes as the key means two band-slices collide ONLY if every one of
    their r values is identical — exactly the AND-within-a-band semantics we proved.
    """
    if signature.shape[0] != b * r:                     # n must equal b*r; fail loud.
        raise ValueError(f"signature length {signature.shape[0]} != b*r = {b*r}")
    sig = np.ascontiguousarray(signature, dtype=np.uint64)  # stable byte layout.
    for band_index in range(b):
        start = band_index * r                          # this band's first row...
        band = sig[start:start + r]                     # ...slice of exactly r rows.
        yield band_index, band.tobytes()                # raw bytes = exact r-tuple key.
```

```python
"""Reference implementation of dedup_pipeline.lsh.bucket_index.BucketIndex."""
from collections import defaultdict


class BucketIndex:
    """Inverted index: (band_index, band_key) -> list of doc ids.

    We key on (band_index, band_key) — NOT band_key alone — so that identical
    r-tuples appearing in DIFFERENT bands never share a bucket. Without the band
    index in the key, band 0 and band 5 of two unrelated docs could falsely collide.
    """

    def __init__(self) -> None:
        # defaultdict(list) auto-creates an empty list on first insert into a bucket.
        self._index: dict[tuple[int, bytes], list[int]] = defaultdict(list)

    def add(self, doc_id: int, band_index: int, band_key: bytes) -> None:
        # Append this doc to the bucket it landed in for this band.
        self._index[(band_index, band_key)].append(doc_id)

    def buckets(self):
        # Iterate only buckets with >= 2 docs — singletons can yield no pair.
        for key, doc_ids in self._index.items():
            if len(doc_ids) >= 2:
                yield doc_ids
```

```python
"""Reference implementation of dedup_pipeline.lsh.candidate_pairs."""
from itertools import combinations
from typing import Iterable, Iterator
import numpy as np

from banding import iter_bands           # the module shown above
from bucket_index import BucketIndex


class _BloomFilter:
    """Tiny Bloom filter to suppress re-emitting a pair seen in multiple bands."""

    def __init__(self, m_bits: int, k: int) -> None:
        self._m = m_bits                                  # bit-array width (eq. 7's m).
        self._k = k                                       # number of hashes (eq. 7's k).
        self._bits = bytearray((m_bits + 7) // 8)         # 8 bits per byte; round up.

    def _positions(self, item: int) -> Iterator[int]:
        # Derive k pseudo-independent positions from one 64-bit value via mixing.
        h = item & ((1 << 64) - 1)
        for i in range(self._k):
            h = (h * 1099511628211 + 0x9E3779B97F4A7C15 + i) & ((1 << 64) - 1)  # mix.
            yield h % self._m                             # fold into the bit array.

    def add_if_absent(self, item: int) -> bool:
        # Return True the FIRST time we see `item`; False on (probable) repeats.
        seen = True
        for pos in self._positions(item):
            byte, bit = divmod(pos, 8)
            if not (self._bits[byte] >> bit) & 1:         # any unset bit => truly new.
                seen = False
                self._bits[byte] |= (1 << bit)            # set it for next time.
        return not seen


def generate_candidate_pairs(
    signatures: np.ndarray, b: int, r: int,
) -> Iterator[tuple[int, int]]:
    """Yield unique (a, b) candidate pairs with a < b from a signature matrix.

    signatures: shape (n_docs, n) uint64 matrix; row i is document i's signature.
    """
    index = BucketIndex()
    for doc_id in range(signatures.shape[0]):             # bucket every document...
        for band_index, band_key in iter_bands(signatures[doc_id], b, r):
            index.add(doc_id, band_index, band_key)       # ...into all b of its bands.

    n_docs = signatures.shape[0]
    bloom = _BloomFilter(m_bits=max(64, 10 * n_docs * b), k=7)  # ~10 bits/elem, k=7.
    for doc_ids in index.buckets():                       # only buckets with >=2 docs.
        for a, c in combinations(sorted(set(doc_ids)), 2): # all within-bucket pairs.
            lo, hi = (a, c) if a < c else (c, a)          # enforce a < b ordering.
            pair_id = lo * n_docs + hi                    # bijective pair -> int key.
            if bloom.add_if_absent(pair_id):              # emit only the first sighting.
                yield (lo, hi)


if __name__ == "__main__":
    # Three near-identical docs (0,1,2) and one outlier (3). r=2, b=2, n=4.
    sigs = np.array([
        [11, 22, 33, 44],   # doc 0
        [11, 22, 33, 99],   # doc 1: shares band 0 (rows 11,22) with doc 0
        [11, 22, 77, 44],   # doc 2: shares band 0 with 0&1; band 1 (33,44) with doc 0
        [55, 66, 88, 90],   # doc 3: shares nothing
    ], dtype=np.uint64)
    print(sorted(generate_candidate_pairs(sigs, b=2, r=2)))
    # -> [(0, 1), (0, 2), (1, 2)]   ; doc 3 is correctly never a candidate.
```

The closing example demonstrates the whole flow end to end: bucketing, the AND/OR
logic, `a < b` ordering, and Bloom-deduplicated emission — with a known-correct
expected output you can run and verify.

---

## 4. Common pitfalls

### Pitfall 1 — "Hashing the band to a 32-bit int" collisions

**Diagnosis.** Engineers often hash each `r`-tuple to a fixed-width integer bucket id
(e.g. `hash(tuple) % 2³²`) to save memory. Two *different* `r`-tuples can then map to
the *same* bucket id, manufacturing candidate pairs that share no rows at all. You
see this as an unexpectedly high false-positive rate that your exact-Jaccard
verifier rejects, plus dedup-rate inflation if you skip verification. With `n_docs·b`
band-slices hashed into `M` buckets, expected spurious collisions grow like
`(n_docs·b)² / (2M)`.

**Fix.** Key buckets on the *raw `r`-tuple bytes* (as in `iter_bands` above), so a
bucket collision requires *every* value to match — true AND semantics. If you must
compress to an int, use ≥ 64-bit keys and *always* run an exact verification pass on
emitted pairs before treating them as duplicates.

### Pitfall 2 — Forgetting the band index in the bucket key

**Diagnosis.** You key buckets on `band_key` alone instead of
`(band_index, band_key)`. Now an `r`-tuple in band 0 of doc A collides with the same
`r`-tuple in band 7 of doc B, even though banding's independence argument assumed
each band is its *own* hash space. Symptom: recall looks suspiciously high and the
candidate count balloons, especially on corpora with repeated boilerplate (license
headers, navigation text) that produces identical band slices in different positions.

**Fix.** Always include `band_index` in the key: `(band_index, band_key)`. Each band
is a separate, disjoint bucket namespace — that is what equation (4)'s independence
across bands requires.

### Pitfall 3 — Choosing `(b, r)` from a memorized threshold instead of recomputing

**Diagnosis.** You pick `(b, r)` because a table or a half-remembered constant said
"that's threshold 0.8," then your dedup rate comes out far higher or lower than
expected. As shown in §2.4, `(b=16, r=8)` is frequently *misquoted* (threshold
0.6866 instead of the correct 0.7071; `P(s=0.8)=0.9997` instead of 0.9471). The
operating point is exponentially sensitive to `b`, so a small misremembering shifts
recall by tens of percent.

**Fix.** Recompute for *your* `(b, r)` every time: threshold from
`t ≈ (1/b)^(1/r)` (eq. 6) and the actual catch probability from
`P = 1 − (1 − s^r)^b` (eq. 5) at the specific `s` values you care about. Tabulate
`P` at `s = t−0.1, t, t+0.1` and confirm the curve crosses where you intend before
committing the run.

### Pitfall 4 (bonus) — Treating the Bloom filter as exact

**Diagnosis.** You use a Bloom filter to deduplicate *emitted pairs* and quietly lose
~`p` of your true pairs (0.82% in the §2.6 sizing) because a false positive reads as
"already emitted, drop it." On a recall-critical run this silently lowers recall and
is hard to spot — the pairs were simply never produced.

**Fix.** Use the Bloom filter only where a small recall loss is acceptable, size it
from equation (7) for your target `p`, and for recall-critical stages fall back to an
exact hash set or emit-then-dedup-at-verify rather than dropping on a probable repeat.

---

## 5. Chapter summary

MinHash shrinks each document to a compact signature but leaves the *comparison*
quadratic — about `5 × 10¹¹` pairs and ~14 core-hours for just 1M documents, growing
with the square of the corpus. **LSH banding** breaks the wall: split the length-`n`
signature into `b` bands of `r` rows (`n = b·r`), bucket each band's `r`-tuple, and
call any two documents that share a bucket in *any* band a candidate. The catch
probability is `P = 1 − (1 − s^r)^b`, derived as "agree on all `r` rows of a band"
(`s^r`) complemented and ANDed across `b` independent bands. This yields the S-curve
with threshold `t ≈ (1/b)^(1/r)`; for `(b=16, r=8)` that is exactly `2^(−1/2) ≈ 0.7071`
and `P(s=0.8) = 0.9471`. More bands raise recall and lower precision; more rows
sharpen the curve. You implement it with an **inverted bucket index keyed on
`(band_index, band_key)`**, deduplicate emitted pairs with a **Bloom filter** sized by
`p ≈ (1 − e^(−kn/m))^k` (≈ 0.82% at 12.5 MB for 10M pairs), and **distribute by band**
so only thin band slices and a compact final pair list ever cross the network. The
recurring discipline: recompute your operating point from the formulas — never trust
a memorized threshold.

---

## 6. Self-check quiz

**Q1.** A corpus has 2,000,000 documents. How many signature pairs would a brute-force
all-pairs MinHash comparison examine, and why does LSH avoid the quadratic blowup?

> **Answer.** Pairs = `2,000,000 · 1,999,999 / 2 = 1,999,999,000,000 ≈ 2.0 × 10¹²`.
> LSH never enumerates all pairs: it buckets each document's band slices and emits
> pairs only *within* shared buckets, so it touches roughly the number of truly
> similar pairs plus a small false-positive margin, not `O(n_docs²)`.

**Q2.** For `b = 20`, `r = 5` (`n = 100`), compute the approximate threshold and the
exact `P(candidate)` at `s = 0.7`.

> **Answer.** Threshold `t = (1/20)^(1/5) = e^((ln 0.05)/5) = e^(−2.9957/5) =
> e^(−0.59915) ≈ 0.5493`. Catch probability: `0.7⁵ = 0.16807`, so
> `P = 1 − (1 − 0.16807)²⁰ = 1 − 0.83193²⁰ = 1 − e^(20·ln 0.83193)
> = 1 − e^(−3.6810) = 1 − 0.02521 ≈ 0.9748`. So at `s = 0.7` the pair is caught with
> ~97.5% probability, consistent with the threshold sitting near 0.55 (well below 0.7).

**Q3.** You key your bucket index on the raw `r`-tuple bytes but *omit* the band
index. What failure mode appears, and what is the one-line fix?

> **Answer.** Identical `r`-tuples in *different* bands collide (e.g. boilerplate text
> producing the same slice in band 0 of doc A and band 7 of doc B), inflating the
> candidate count and recall and violating the cross-band independence assumption.
> Fix: key on `(band_index, band_key)` so each band is its own bucket namespace.

---

## References

- **[Bloom 1970]** B. H. Bloom. "Space/Time Trade-offs in Hash Coding with Allowable
  Errors." *Communications of the ACM*, 13(7):422–426, 1970.
- **[Broder 1997]** A. Z. Broder. "On the Resemblance and Containment of Documents."
  *Proceedings of the Compression and Complexity of Sequences (SEQUENCES)*, 1997.
- **[Indyk & Motwani 1998]** P. Indyk and R. Motwani. "Approximate Nearest Neighbors:
  Towards Removing the Curse of Dimensionality." *Proceedings of the 30th ACM
  Symposium on Theory of Computing (STOC)*, pp. 604–613, 1998.
- **[Leskovec et al. 2014]** J. Leskovec, A. Rajaraman, and J. D. Ullman. *Mining of
  Massive Datasets (MMDS)*, 2nd ed., Chapter 3 ("Finding Similar Items"). Cambridge
  University Press, 2014.
