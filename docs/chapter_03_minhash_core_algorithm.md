# Chapter 3: MinHash — The Core Algorithm

In Chapter 2 we turned documents into **sets of integer shingles** (overlapping
k-grams hashed to integers) and defined the **Jaccard similarity** of two sets
`A` and `B` as

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

That formula is exactly right and completely impractical. A web document might
contain 5,000 distinct shingles; a corpus might contain 100 million documents.
Storing the full shingle sets and intersecting every pair is hopeless on both
memory and compute. MinHash is the trick that lets us replace each multi-thousand
element set with a tiny fixed-length integer *signature* — say 128 numbers —
such that the **fraction of matching signature positions is an unbiased estimate
of the Jaccard similarity** [Broder 1997]. This chapter derives that result from
first principles, analyzes its accuracy, and builds the production implementation
that lives in `dedup_pipeline.minhash`.

---

## 1. Learning objectives

After this chapter you will be able to:

- **Define a universal hash family** and construct one with the classic
  `h(x) = ((a·x + b) mod p) mod m` form, explaining why each hash acts as a
  random permutation of the shingle universe [Carter & Wegman 1979].
- **Prove** that for a single random permutation,
  `P(min(h(A)) == min(h(B))) = J(A, B)` — i.e. one bit is an unbiased estimator
  of Jaccard similarity [Broder 1997].
- **Derive the variance** `Var(Ĵ) = J(1−J)/n` of the `n`-hash estimator and
  use it to choose `n` for a target standard error.
- **Implement MinHash four ways** (naive Python → vectorized NumPy → batched
  broadcasting → optional Numba) and reason about the speed/memory trade-offs.
- **Lay out the signature matrix** as a C-contiguous `uint32` array of shape
  `(n_docs, n_hashes)` and compute its memory cost.

---

## 2. Concept explanation

### 2.1 Hash families and universal hashing

A **hash function** maps a key from a large *universe* `U` (here, the set of all
possible shingle integers, e.g. all 64-bit values) into a smaller range
`{0, 1, …, m−1}`. A **single fixed** hash function is useless for our purposes:
two different documents would always collide or not collide in exactly the same
way, giving us no randomness to average over. We need a *family* of hash
functions and the ability to draw a fresh random one.

A family `H` is **universal** if, for any two distinct keys `x ≠ y`, a randomly
chosen `h ∈ H` collides with probability at most `1/m`:

```
Pr_{h ∈ H}[ h(x) == h(y) ] ≤ 1/m
```

[Carter & Wegman 1979] gave the construction we use everywhere in this pipeline.
Pick a prime `p` larger than the universe size, draw `a ∈ {1, …, p−1}` and
`b ∈ {0, …, p−1}` uniformly at random, then define

```
h_{a,b}(x) = ((a·x + b) mod p) mod m
```

The `(a·x + b) mod p` part is a **bijection on `{0, …, p−1}`** (because `a ≠ 0`
and `p` is prime, so `a` is invertible mod `p`). A bijection is just a
*permutation*: as `(a, b)` range over their domains, the term shuffles the
universe into a new order. **This is the conceptual heart of MinHash.** Each
hash function imposes a *random permutation* of the shingle universe, and asking
"which shingle hashes to the smallest value?" is the same as asking "which
shingle comes *first* under this random ordering?"

In practice we set `m` to the full `uint32` (or `uint64`) range so the second
`mod m` is essentially free and collisions among the actual shingles are rare.
The pipeline's `dedup_pipeline.minhash.hash_functions` module stores the
`(a, b)` pairs as two integer arrays and can optionally substitute a fast
non-cryptographic hash (xxhash) for the permutation, which behaves like a
member of a min-wise independent family for our accuracy needs
[Broder et al. 1998].

### 2.2 The MinHash estimator: the core proof

Fix one random permutation `π` of the universe (one hash function `h`). For a
set `S`, define its **MinHash value** as

```
h_min(S) = min_{x ∈ S} h(x)
```

i.e. the element of `S` that lands first under `π`. The central claim:

> **Theorem (Broder 1997).** For a single random permutation,
> `Pr[ h_min(A) == h_min(B) ] = J(A, B)`.

**Proof.** Consider the union `A ∪ B`. Look at the element of `A ∪ B` whose hash
is globally smallest — call it `x*`, the *argmin* over the union. Because `π` is
a uniformly random permutation, **`x*` is equally likely to be any of the
`|A ∪ B|` elements**; no element is privileged.

Now ask: when do the two minima agree, `h_min(A) == h_min(B)`? The minimum over
`A` is the first `A`-element under `π`; the minimum over `B` is the first
`B`-element under `π`. The globally-first element of the union, `x*`, is
necessarily the minimum of *whichever sets contain it*. Two cases:

- If `x* ∈ A ∩ B`, then `x*` is simultaneously the minimum of `A` and the
  minimum of `B`, so `h_min(A) == h_min(B) == h(x*)`. **The minima match.**
- If `x*` lies in exactly one set (in `A \ B` or `B \ A`), then `x*` is the
  minimum of that set but the *other* set's minimum is some later element, so
  the two minima differ. **The minima do not match.**

(The argmin cannot lie outside `A ∪ B`, and ties have probability zero for a
genuine permutation.) Therefore the minima match **if and only if** `x*`, the
uniformly-random argmin, happens to fall in `A ∩ B`:

```
Pr[ h_min(A) == h_min(B) ] = |A ∩ B| / |A ∪ B| = J(A, B).   ∎
```

The remarkable consequence: define the indicator `X = 1{ h_min(A) == h_min(B) }`.
Then `E[X] = J(A, B)`. **A single bit — do the two minima match? — is an
unbiased estimator of the Jaccard similarity.** We did not need to store the
sets, intersect them, or count anything. We compared two integers.

### 2.3 Variance and how many hashes you need

One bit has `E[X] = J`, but its variance is enormous: a Bernoulli(`J`) variable
has variance `J(1−J)`, so a single hash gives you a coin flip, not a measurement.
The fix is averaging. Draw `n` **independent** hash functions
`h_1, …, h_n` and form the estimator

```
Ĵ = (1/n) · Σ_{i=1}^{n} 1{ h_min^(i)(A) == h_min^(i)(B) }
```

This is the **fraction of signature positions that match**. Each term is an
independent Bernoulli(`J`) trial, so:

```
E[Ĵ]   = J                          (unbiased)
Var(Ĵ) = J(1 − J) / n
SE(Ĵ)  = sqrt( J(1 − J) / n )
```

Because `J(1−J)` is maximized at `J = 0.5` with value `0.25`, the standard error
is bounded for *every* `J`:

```
SE(Ĵ) ≤ sqrt(0.25 / n) = 1 / (2·sqrt(n))
```

The error shrinks like `1/sqrt(n)` — the classic Monte-Carlo rate. To halve the
error you must *quadruple* the number of hashes. Concrete numbers for `J = 0.8`
(so `J(1−J) = 0.16`):

| `n` (hashes) | `Var(Ĵ) = 0.16/n` | `SE = sqrt(Var)` | 2-sigma band `±2·SE` |
|--------------|-------------------|------------------|----------------------|
| 64           | 0.002500          | **0.0500**       | ±0.100               |
| 128          | 0.001250          | **0.0354**       | ±0.071               |
| 256          | 0.000625          | **0.0250**       | ±0.050               |

So at the pipeline default of `n = 128`, a true Jaccard of 0.80 is estimated to
within about `±0.035` one standard error, `±0.071` at two sigma. Note the
diminishing returns: going `64 → 128` buys a `0.050 → 0.035` improvement, while
`128 → 256` only buys `0.035 → 0.025`. The error-vs-`n` curve is a slowly
flattening `1/sqrt(n)` decay, which is why production systems cluster around
`128–256` hashes rather than pushing into the thousands — the cost is linear in
`n` but the accuracy payoff is sub-linear [Leskovec et al. 2014]. This bound is
also why the pipeline's acceptance test allows an error of `2/sqrt(n)` (the
two-sigma worst case at `J = 0.5`).

### 2.4 A fully worked numerical example

Let us do the whole computation by hand. Take two tiny documents and their
3-shingle integer IDs (assume Chapter 2's shingler produced these IDs):

```
Doc A shingles: {10, 17, 23, 31, 42}
Doc B shingles: {17, 23, 31, 50, 64}
```

The intersection is `{17, 23, 31}` (3 elements) and the union is
`{10, 17, 23, 31, 42, 50, 64}` (7 elements), so the **true Jaccard** is

```
J(A, B) = 3 / 7 = 0.4286
```

Now apply `n = 8` explicit hash functions `h_i(x) = (a_i·x + b_i) mod p` with the
prime `p = 97` (any prime above our shingle IDs works for this toy). The
`(a, b)` pairs:

| i | a | b |
|---|---|---|
| 1 |  3 |  7 |
| 2 |  5 | 11 |
| 3 |  7 |  1 |
| 4 | 11 |  4 |
| 5 | 13 |  9 |
| 6 | 17 |  2 |
| 7 | 19 | 15 |
| 8 | 23 |  6 |

For each hash, compute `h_i(x) mod 97` for every shingle in a document and take
the minimum. Hash 1, `h(x) = (3x + 7) mod 97`:

```
A: h(10)=37, h(17)=58, h(23)=76, h(31)=3,  h(42)=36  → min = 3   (from shingle 31)
B: h(17)=58, h(23)=76, h(31)=3,  h(50)=60, h(64)=2   → min = 2   (from shingle 64)
```

Minima differ → **no match**. Working through all 8 hashes gives:

| i | (a, b)  | min(h(A)) | min(h(B)) | match? |
|---|---------|-----------|-----------|--------|
| 1 | (3, 7)  | 3   (x=31)| 2   (x=64)| no     |
| 2 | (5, 11) | 27  (x=42)| 29  (x=23)| no     |
| 3 | (7, 1)  | 4   (x=42)| 23  (x=17)| no     |
| 4 | (11, 4) | 17  (x=10)| 29  (x=64)| no     |
| 5 | (13, 9) | 17  (x=23)| 17  (x=23)| **yes**|
| 6 | (17, 2) | 0   (x=17)| 0   (x=17)| **yes**|
| 7 | (19, 15)| 11  (x=10)| 22  (x=31)| no     |
| 8 | (23, 6) | 2   (x=42)| 9   (x=17)| no     |

Matches: **2 out of 8** (hashes 5 and 6, where the union's first element is the
shared shingle 23 and 17 respectively). The MinHash estimate is

```
Ĵ = 2 / 8 = 0.250
```

versus the true `J = 0.4286`. The absolute error is
`|0.250 − 0.4286| = 0.179`. That gap is *exactly* what the variance formula
predicts for such a small `n`: at `J ≈ 0.43`,
`SE = sqrt(0.4286·0.5714/8) = sqrt(0.0306) ≈ 0.175`, so an error of `0.179` is
almost exactly **one standard error** — a completely ordinary outcome. With only
8 coin flips, deviations this size are routine. This is the entire reason
production uses `n = 128`: scale `n` up and the estimate collapses onto the true
value. (You can reproduce every entry by evaluating `(a·x + b) mod 97` for each
shingle and taking the column minimum.)

---

## 3. Annotated code walkthrough

We build the implementation in four stages, each faster than the last. All code
is runnable with `numpy` plus the standard library; the final stage optionally
uses `numba`. These mirror `dedup_pipeline.minhash.minhash.MinHasher`.

```python
import numpy as np

# A Mersenne prime > 2**32, so it exceeds any uint32 shingle ID. Using a prime
# guarantees (a*x + b) mod p is a permutation (a is invertible mod p).
MERSENNE_P = (1 << 61) - 1          # 2305843009213693951, a 61-bit prime
MAX_HASH   = (1 << 32) - 1          # we fold results into uint32 range


def make_hashes(n_hashes: int, seed: int = 0):
    """Draw n universal-hash parameters (a, b). This is MinHasher.fit()."""
    rng = np.random.default_rng(seed)            # reproducible: same seed → same hashes
    a = rng.integers(1, MERSENNE_P, size=n_hashes, dtype=np.uint64)  # a ∈ [1, p-1], never 0
    b = rng.integers(0, MERSENNE_P, size=n_hashes, dtype=np.uint64)  # b ∈ [0, p-1]
    return a, b
```

### Stage 1 — naive pure-Python double loop (the reference)

```python
def signature_naive(shingles, a, b):
    """shingles: 1-D array of int shingle IDs for ONE document.
    Returns a length-n_hashes uint32 signature."""
    n_hashes = len(a)
    sig = np.empty(n_hashes, dtype=np.uint32)
    for i in range(n_hashes):                    # outer loop: one per hash function
        m = MAX_HASH
        for x in shingles:                       # inner loop: one per shingle
            # Python ints are arbitrary precision, so no overflow here.
            hv = ((int(a[i]) * int(x) + int(b[i])) % MERSENNE_P) & MAX_HASH
            if hv < m:                            # running minimum
                m = hv
        sig[i] = m
    return sig
```

This is `O(n_hashes · n_shingles)` with full Python interpreter overhead on every
operation. For 128 hashes and 500 shingles that is 64,000 pure-Python modulo
operations per document — roughly **2–4 ms/doc**, i.e. only a few hundred
docs/sec. Correct, but unusably slow. Keep it: it is the oracle your fast
versions must match exactly.

### Stage 2 — vectorized NumPy, one hash at a time

```python
def signature_vectorized(shingles, a, b):
    x = shingles.astype(np.uint64)               # promote once to avoid overflow in a*x
    sig = np.empty(len(a), dtype=np.uint32)
    for i in range(len(a)):                       # still loop over hashes...
        # ...but the inner shingle loop is now a single vectorized expression:
        hv = ((a[i] * x + b[i]) % MERSENNE_P) & MAX_HASH   # whole array at once, in C
        sig[i] = hv.min()                          # NumPy reduction, also in C
    return sig
```

We deleted the inner Python loop; NumPy applies the affine map and the `min`
reduction across all shingles in compiled C. Same arithmetic, same result,
roughly **30–50x faster** (~50–80 µs/doc) because the interpreter no longer
touches each shingle. Memory is tiny: one temporary array of `n_shingles`.

### Stage 3 — batched broadcasting over the full matrix

```python
def signature_batched(shingles, a, b):
    x = shingles.astype(np.uint64)               # shape (S,)
    A = a.reshape(-1, 1)                          # shape (H, 1)
    B = b.reshape(-1, 1)                          # shape (H, 1)
    # Broadcasting (H,1) * (S,) → (H, S): every hash applied to every shingle.
    H = ((A * x + B) % MERSENNE_P) & MAX_HASH     # shape (H, S), one big tensor op
    sig = H.min(axis=1)                           # column-wise (per-hash) minimum → (H,)
    return sig.astype(np.uint32)
```

Now *both* loops are gone — the entire `(n_hashes × n_shingles)` matrix is
computed in one broadcasted expression, and `min(axis=1)` collapses each row to
its minimum. This is the fastest *pure-NumPy* form, ~**2–5x faster than Stage 2**
(~15–25 µs/doc), because it has zero Python-level iteration. The trade-off is
**memory**: it materializes an `H × S` array. For `H = 128` and `S = 500` that
is 64,000 `uint64` values (~512 KB) — fine. But a pathological 50,000-shingle
document would need `128 × 50,000 × 8 = 51 MB` transiently. Production code caps
the matrix by processing very long documents in shingle-chunks, taking a running
column-min across chunks.

### Stage 4 — optional Numba JIT inner loop

```python
try:
    from numba import njit, prange

    @njit(parallel=True, fastmath=False, cache=True)
    def signature_numba(shingles, a, b, p, mask):
        H = a.shape[0]
        sig = np.empty(H, dtype=np.uint64)
        for i in prange(H):                       # prange → threads run hashes in parallel
            m = mask                               # init running min to MAX_HASH
            ai = a[i]; bi = b[i]
            for j in range(shingles.shape[0]):     # tight scalar loop, compiled to machine code
                hv = ((ai * shingles[j] + bi) % p) & mask
                if hv < m:
                    m = hv
            sig[i] = m
        return sig
except ImportError:
    signature_numba = None
```

Numba compiles the double loop to native code and runs the hash dimension across
CPU threads with `prange`. Unlike Stage 3 it **never materializes the `H × S`
matrix** — it keeps a single scalar running-min per hash, so memory is `O(H)`
regardless of document length. On a multi-core box it reaches **~5–10 µs/doc**
and scales linearly with cores. Rough relative throughput on one document of 500
shingles, 128 hashes:

| Stage | Method                | Time/doc | Speedup vs naive | Peak extra memory |
|-------|-----------------------|----------|------------------|-------------------|
| 1     | Naive Python          | ~3 ms    | 1x               | O(1)              |
| 2     | Vectorized (per-hash) | ~65 µs   | ~45x             | O(S)              |
| 3     | Batched broadcasting  | ~20 µs   | ~150x            | O(H·S)            |
| 4     | Numba JIT (parallel)  | ~7 µs    | ~430x            | O(H)              |

(Numbers are representative on a modern x86 core; absolute values vary, but the
*ordering* and the memory column are the stable lessons.) `MinHasher.transform`
selects Stage 3 by default and Stage 4 when Numba is importable;
`MinHasher.batch_transform` loops the chosen kernel over a list of documents and
writes rows directly into the preallocated signature matrix.

### The signature matrix and `signature_store`

```python
class MinHasher:
    def __init__(self, n_hashes=128, seed=0):
        self.n_hashes = n_hashes
        self.a = None
        self.b = None
        self.seed = seed

    def fit(self):                                # draw and freeze the hash family
        self.a, self.b = make_hashes(self.n_hashes, self.seed)
        return self

    def transform(self, shingles):                # one doc → one signature row
        return signature_batched(np.asarray(shingles), self.a, self.b)

    def batch_transform(self, docs):              # many docs → (n_docs, n_hashes) matrix
        n = len(docs)
        # Preallocate row-major (C-contiguous) uint32: rows = documents.
        sig = np.empty((n, self.n_hashes), dtype=np.uint32)
        for r, shingles in enumerate(docs):
            sig[r, :] = self.transform(shingles)   # fill one row per document
        return sig
```

The **signature matrix** has shape `(n_docs, n_hashes)`, dtype `uint32`, stored
**C-contiguous (row-major)** so each document's signature is a contiguous slice —
exactly the access pattern the LSH banding of Chapter 4 wants. Memory is
`n_docs · n_hashes · 4` bytes:

```
1,000,000 docs × 128 hashes × 4 bytes = 512,000,000 ≈ 512 MB
```

`uint32` is the right default: a 32-bit range gives a collision probability of
roughly `S²/2³³` per hash, negligible for realistic shingle counts, and it halves
storage versus `uint64` (which would be **1 GB** for the same corpus).
`dedup_pipeline.minhash.signature_store` provides both an in-memory `uint32`
array and an mmap-backed variant so a 100 GB signature set can live on disk and
be paged in band-by-band without ever fitting in RAM.

---

## 4. Common pitfalls

### Pitfall 1 — **Non-prime modulus (the broken permutation)**

**Diagnosis.** Someone replaces the prime `p` with a power of two (e.g.
`& 0xFFFFFFFF` *as the only reduction*, with no `mod p`), or picks a composite
`p`. The map `a·x + b mod p` is then **not a bijection**: distinct shingles
collapse onto the same hash value far more often than `1/m`, the "permutation"
develops fixed structure, and Jaccard estimates become biased — typically
inflated, because spurious collisions force more minima to agree. You will see
the acceptance test `|Ĵ − J| ≤ 2/sqrt(n)` fail systematically on dissimilar
documents.

**Fix.** Always reduce modulo a **prime strictly greater than the universe
size**, with `a ∈ [1, p−1]` (never `a = 0`, which makes `h` constant). Use the
Mersenne prime `2⁶¹ − 1` and fold to `uint32` *after* the prime reduction, as in
the code above. The `mod m` fold is a separate, final step.

### Pitfall 2 — **Integer overflow in `a·x`**

**Diagnosis.** The hash is computed in `uint32` or `int32`. The product `a·x`
overflows silently (NumPy `uint32` wraps modulo `2³²`, C-level Numba wraps too),
so `(a·x + b) mod p` is computed on a corrupted product. Symptoms: signatures are
*reproducible* (so tests that only check determinism pass) but *wrong* — two
identical documents still match perfectly, yet near-duplicates score far too low,
because the wraparound scrambles the ordering inconsistently across documents.

**Fix.** Promote shingles and `(a, b)` to `uint64` (or do the multiply in Python
arbitrary-precision in the naive reference). `a` and `x` are up to ~32 bits, so
`a·x` needs ~64 bits — `uint64` holds it before the `mod p`. In Numba, declare
the arrays `uint64` explicitly. Always validate the fast kernels against the
naive Stage-1 oracle on random inputs.

### Pitfall 3 — **Too few hashes, then blaming the algorithm**

**Diagnosis.** An engineer runs MinHash with `n = 16` or `n = 32`, sees
estimates swinging ±0.15 around the truth (just like our worked example's 0.32
error at `n = 8`), and concludes "MinHash is inaccurate." The estimator is
unbiased; the *single-sample variance* `J(1−J)/n` is simply large for small `n`.
At `n = 16` the standard error at `J = 0.5` is `1/(2·4) = 0.125`.

**Fix.** Choose `n` from the variance formula for your *target* precision: solve
`sqrt(J(1−J)/n) ≤ ε` for `n`. For `ε = 0.025` worst-case, `n ≥ 1/(2ε)² = 400`,
or use the corpus default of `128` for `±0.044` worst-case. Remember the cost is
linear in `n` but accuracy improves only as `1/sqrt(n)` — do not expect to fix a
noisy pipeline by nudging `n` from 128 to 160.

### Pitfall 4 (bonus) — **Reusing the same seed across runs you want to compare**

**Diagnosis.** Two corpora are MinHashed with *different* seeds, then their
signatures are compared position-by-position. Because each `MinHasher` drew a
*different* random permutation set, matching signatures is meaningless — you are
comparing apples under permutation π against oranges under permutation σ.

**Fix.** The hash family is part of the model. `MinHasher.fit(seed=...)` must use
the **same seed** for every document set you intend to compare, and the
`(a, b)` arrays should be serialized alongside the signature store so future runs
stay compatible.

---

## 5. Chapter summary

MinHash replaces an unwieldy shingle set with a short integer signature whose
matching rate estimates Jaccard similarity. The mechanism is a **universal hash
family** [Carter & Wegman 1979], `h(x) = ((a·x + b) mod p) mod m` with `p` a
prime above the universe size — each hash is a random *permutation* of the
shingle universe. For a single permutation, the minima of two sets agree exactly
when the globally-first union element lies in the intersection, which happens with
probability `|A ∩ B| / |A ∪ B| = J(A, B)`; thus **one bit is an unbiased
estimator of Jaccard** [Broder 1997]. Averaging `n` independent hashes gives
`Ĵ` with `E[Ĵ] = J` and `Var(Ĵ) = J(1−J)/n`, a standard error bounded by
`1/(2·sqrt(n))` — the `1/sqrt(n)` Monte-Carlo rate that motivates `n = 128–256`
in practice [Leskovec et al. 2014]. Implementation progresses from a naive
double loop to batched NumPy broadcasting to an optional Numba kernel, trading
`O(H·S)` transient memory for speed, and the result is a C-contiguous
`(n_docs, n_hashes)` `uint32` signature matrix (512 MB for a million docs at 128
hashes) managed by `dedup_pipeline.minhash.signature_store`.

We now have signatures. But comparing all `n_docs²` signature pairs is still
quadratic — Chapter 4 introduces **Locality-Sensitive Hashing** to find candidate
near-duplicates in near-linear time.

---

## 6. Self-check quiz

**Q1.** You MinHash with `n = 64` hash functions and measure `Ĵ = 0.50` for a
document pair. Roughly how wide is the one-standard-error band on the true
Jaccard, and what is the cheapest principled way to halve it?

> **Answer.** At `J ≈ 0.5`, `SE = sqrt(0.5·0.5/64) = sqrt(0.00390625) = 0.0625`.
> To halve the standard error you must **quadruple** `n` (since `SE ∝ 1/sqrt(n)`),
> so go from 64 to 256 hashes, giving `SE = sqrt(0.25/256) = 0.03125`. There is no
> cheaper way within MinHash — the `1/sqrt(n)` rate is fundamental to averaging
> independent Bernoulli trials.

**Q2.** A teammate sets the modulus to `p = 2³²` (a power of two) "to make the
fold free" and reports that dissimilar documents now look 10–15% more similar
than expected. Name the bug and the one-line fix.

> **Answer.** `2³²` is **not prime**, so `(a·x + b) mod 2³²` is not a bijection —
> it is not a valid permutation, and spurious collisions inflate the match rate
> (Pitfall 1). Fix: reduce modulo a true prime above the universe size (e.g. the
> Mersenne prime `2⁶¹ − 1`), and only *then* fold into `uint32` with a separate
> `& 0xFFFFFFFF`.

**Q3.** In the worked example we got `Ĵ = 2/8 = 0.25` while the true Jaccard was
`3/7 = 0.4286`, an error of 0.18. Is this evidence that the estimator is biased?
Justify with a number.

> **Answer.** No — the estimator is **unbiased** (`E[Ĵ] = J` exactly); the
> error is *variance* from using only `n = 8` samples. At `J ≈ 0.4286`,
> `SE = sqrt(0.4286·0.5714/8) = sqrt(0.0306) ≈ 0.175`, so an observed error of
> `0.18` is essentially one standard error — entirely consistent with an unbiased
> estimator that simply has high variance at small `n`. Averaging many such
> 8-hash estimates would converge to `0.4286`.

---

## References

- [Broder 1997] A. Z. Broder. *On the Resemblance and Containment of Documents.*
  Proceedings of the Compression and Complexity of Sequences (SEQUENCES), 1997.
  (Introduces MinHash / resemblance via min-wise permutations.)
- [Broder et al. 1998] A. Z. Broder, M. Charikar, A. M. Frieze, M. Mitzenmacher.
  *Min-Wise Independent Permutations.* Proceedings of the 30th ACM Symposium on
  Theory of Computing (STOC), 1998.
- [Carter & Wegman 1979] J. L. Carter, M. N. Wegman. *Universal Classes of Hash
  Functions.* Journal of Computer and System Sciences, 18(2):143–154, 1979.
- [Leskovec et al. 2014] J. Leskovec, A. Rajaraman, J. D. Ullman. *Mining of
  Massive Datasets (MMDS), 2nd ed.* Cambridge University Press, 2014.
  (Chapter 3: Finding Similar Items — MinHash and LSH.)
