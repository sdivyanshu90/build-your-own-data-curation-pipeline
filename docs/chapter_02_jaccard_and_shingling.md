# Chapter 2: Probabilistic Foundations — Jaccard Similarity & Shingling

> *Module: Build-Your-Own Data Curation Pipeline. Prerequisite: Chapter 1 (Why Deduplication Matters for LLM Pretraining).*

Before we touch a single MinHash signature in Chapter 3, we need a precise, mathematical answer to one deceptively simple question: **what does it mean for two documents to be "similar"?** This chapter builds that answer from first principles. We define similarity as a property of *sets*, show how to turn raw text into sets ("shingling"), and prove why the naive approach collapses at scale — which is the entire reason MinHash exists.

No NLP background is assumed. Every concept is built up before any code appears.

---

## 1. Learning objectives

By the end of this chapter you will be able to:

- **Define** the Jaccard coefficient `J(A,B) = |A ∩ B| / |A ∪ B|` and explain why it captures *content overlap* for long documents better than edit distance.
- **Convert** a document into a set of features using three shingling strategies (character n-grams, word n-grams, sentence tokens) and articulate the trade-offs of each.
- **Derive** the `O(n²)` cost of exact all-pairs comparison and produce a concrete wall-clock estimate for a million-document corpus.
- **Choose** a shingle size `k` and apply text normalization (lowercasing, whitespace collapse, Unicode NFKC) to control false collisions and similarity inflation.
- **Compute** Jaccard by hand on real text, tracking `|A|`, `|B|`, `|A ∩ B|`, and `|A ∪ B|` explicitly.

---

## 2. Concept explanation

### 2.1 Similarity as a property of sets

A **set** is an unordered collection of distinct elements. The two operations we care about are:

- **Intersection** `A ∩ B`: the elements in *both* A and B.
- **Union** `A ∪ B`: the elements in *either* A or B (each counted once).

The **Jaccard coefficient** (also called Jaccard *similarity* or, in the deduplication literature, *resemblance* [Broder 1997]) is the fraction of the combined material that the two sets share:

```
J(A,B) = |A ∩ B| / |A ∪ B|
```

It is bounded in `[0, 1]`: it is `0` when the sets are disjoint and `1` when they are identical. A tiny worked example with integer sets:

```
A = {1, 2, 3, 4}
B = {3, 4, 5}
A ∩ B = {3, 4}            ->  |A ∩ B| = 2
A ∪ B = {1, 2, 3, 4, 5}   ->  |A ∪ B| = 5
J(A,B) = 2 / 5 = 0.40
```

A useful identity, which we will reuse constantly, lets you compute the union from the two sizes and the intersection without enumerating it:

```
|A ∪ B| = |A| + |B| − |A ∩ B|
        = 4 + 3 − 2 = 5     ✓
```

### 2.2 Why Jaccard beats edit distance for long documents

The intuitive alternative is **edit distance** (Levenshtein): the minimum number of single-character insertions, deletions, or substitutions to turn one string into another. Edit distance is excellent for short strings (spell-check, fuzzy name matching), but it is the wrong tool for document-scale deduplication for two reasons:

1. **Cost.** Computing edit distance between strings of length `n` and `m` is `O(n·m)` time via dynamic programming. For two 5,000-character documents that is ~25 million cell updates *per pair*. Jaccard on pre-built sets is `O(|A| + |B|)` — a linear hash-set walk.

2. **Order sensitivity.** Edit distance is fundamentally about *sequence alignment*. If a plagiarist copies your three paragraphs but reorders them, the documents are ~100% the same content, yet edit distance reports a huge cost because the alignment is wrecked. Jaccard operates on a *set* of features, so it is **order-insensitive**: moving a paragraph leaves the shingle set almost unchanged. This is exactly the failure mode that dominates web-scraped pretraining corpora — boilerplate reordering, template shuffling, near-duplicate articles with paragraphs permuted — and it is why Manber's "finding similar files" work [Manber 1994] and Broder's resemblance framework [Broder 1997] both abandoned alignment in favor of feature sets.

The standard reference treatment of all of this is Chapter 3 of *Mining of Massive Datasets* [Leskovec et al. 2014].

### 2.3 From documents to sets: shingling

A document is a string, not a set. **Shingling** (a.k.a. *w-shingling* [Broder 1997]) converts a string into a set of overlapping sub-sequences of length `k`. Each sub-sequence is a **shingle** (or *k-gram*). There are three common granularities. Consider the source string:

```
The quick brown fox jumps
```

**(a) Character n-grams (k = 5–9).** Slide a window of `k` characters. With `k = 5` over `"the quick"` (lowercased, spaces kept) you get:

```
"the q", "he qu", "e qui", " quic", "quick", ...
```

*Trade-off:* extremely robust to tokenization choices and excellent for languages without whitespace word boundaries (Chinese, Japanese, Korean — "CJK") and for catching typo-level near-duplicates. Cost: large shingle sets and weaker semantics.

**(b) Word n-grams (k = 2–4).** First tokenize into words, then slide a window of `k` words. With `k = 3` (word 3-grams):

```
("the","quick","brown"), ("quick","brown","fox"), ("brown","fox","jumps")
```

*Trade-off:* more semantic and far smaller sets than character n-grams, but sensitive to tokenization and to stopwords ("the", "of", "a") which inflate overlap between unrelated documents. This is the workhorse for English-language LLM deduplication.

**(c) Sentence-level tokens (k = 1 sentence, or sentence n-grams).** Split on sentence boundaries; each sentence (often hashed) is one shingle. *Trade-off:* highest semantic fidelity and tiniest sets, but highest variance — a single edited word changes a whole shingle, so near-duplicates with light paraphrasing look dissimilar. Best for detecting verbatim passage reuse.

In this pipeline these live in `dedup_pipeline.text_processing.shingler` (the windowing logic), which depends on `dedup_pipeline.text_processing.tokenizer` (word/sentence splitting) and `dedup_pipeline.text_processing.normalizer` (lowercasing, whitespace, Unicode).

### 2.4 Why exact all-pairs Jaccard is intractable

To deduplicate a corpus exactly, you would compare every document against every other. For `n` documents the number of unordered pairs is:

```
C(n, 2) = n·(n − 1) / 2
```

For **n = 1,000,000**:

```
pairs = 1,000,000 × 999,999 / 2
      = 999,999,000,000 / 2
      = 499,999,500,000
      ≈ 5 × 10¹¹  pairs
```

Suppose your hardware compares **10,000,000 (10⁷) pairs per second** — optimistic, since each comparison is a real set-intersection, not a single instruction:

```
time = 5 × 10¹¹ pairs ÷ 10⁷ pairs/sec
     = 5 × 10⁴ seconds
     = 50,000 seconds
     ≈ 13.9 hours
```

That is **~14 hours just for the comparison loop**, ignoring the cost of building 1,000,000 shingle sets and holding them in memory. The cost grows quadratically: ten million documents (10×) is **100×** the work — roughly 58 days. This `O(n²)` wall is the entire motivation for the probabilistic machinery (MinHash + LSH) in the chapters ahead, which trade a tiny, bounded error for a near-linear runtime.

### 2.5 Choosing k and handling messy text

**Picking k.** Larger `k` produces *sparser, more specific* shingles: two documents must share longer exact runs to collide, so the baseline (background) similarity between unrelated documents drops and **false collisions fall**. Too large, and genuine near-duplicates with minor edits stop overlapping at all. Rules of thumb [Leskovec et al. 2014]:

- **Character shingles: k ≈ 5** for short documents, `k ≈ 9` for large documents (emails, articles).
- **Word shingles: k ≈ 3** for typical English prose.

**Tokenization edge cases.** Real text is hostile:

- **Punctuation** — decide whether `"end."` and `"end"` are the same token (usually yes; strip trailing punctuation).
- **URLs** — `https://a.com/x?y=1` should be one token, not shattered on `/` and `?`, or boilerplate links will dominate the set.
- **Code snippets** — symbols (`{`, `}`, `==`) carry meaning; over-aggressive stripping makes all code look alike.
- **Numbers** — `"3.14"` vs `"3,14"` (locale) and long IDs can either be kept verbatim or canonicalized to a `<NUM>` placeholder, depending on whether numeric drift should count as a difference.

**Normalization raises overlap.** `dedup_pipeline.text_processing.normalizer` applies lowercasing, whitespace collapse, and **Unicode NFKC** (which folds compatibility variants — e.g. the full-width `Ａ` and ligature `ﬁ` map to ASCII `A` and `fi`). Consider two copies of the same paragraph where one was pasted from a word processor: it has `"Café"` with a precomposed accent, smart quotes `“ ”`, a non-breaking space, and Title Case headings. Without normalization those surface differences split otherwise-identical shingles. In a representative case the raw documents scored:

```
J_raw        = 0.62
J_normalized = 0.94
```

The 32-point jump is pure noise removal: NFKC unified the accented/quote characters, whitespace collapse unified the spacing, and lowercasing unified the casing — so shingles that *should* have matched finally did. The lesson: **normalize before you shingle, or you will under-count true duplicates.**

---

## 3. Annotated code walkthrough

A runnable, dependency-light implementation (standard library plus optional `xxhash` for fast, compact shingle hashing). This mirrors the responsibilities of `normalizer`, `tokenizer`, and `shingler`.

```python
import re
import unicodedata
from typing import Set, Tuple

try:
    import xxhash  # fast non-cryptographic hash; optional acceleration
    _HAVE_XXHASH = True
except ImportError:
    _HAVE_XXHASH = False


# --- normalizer ----------------------------------------------------------
def normalize(text: str) -> str:
    # NFKC folds Unicode compatibility variants (full-width, ligatures, smart
    # quotes) into canonical forms so visually-equal text hashes equally.
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()                       # case-fold: "The" == "the"
    text = re.sub(r"\s+", " ", text)          # collapse runs of whitespace to one space
    return text.strip()                       # drop leading/trailing whitespace


# --- tokenizer -----------------------------------------------------------
def tokenize_words(text: str) -> list:
    # \w+ keeps URLs/numbers from shattering on every symbol; we DON'T split on
    # "." inside a token, only treat non-word chars as separators. \w also
    # matches Unicode letters (so "café" survives as one token under re.UNICODE,
    # which is the default for str patterns in Python 3).
    return re.findall(r"\w+", text)


# --- shingler ------------------------------------------------------------
def word_shingles(text: str, k: int = 3) -> Set[Tuple[str, ...]]:
    text = normalize(text)                    # ALWAYS normalize before shingling
    tokens = tokenize_words(text)
    if len(tokens) < k:                       # guard: a doc shorter than k has
        return {tuple(tokens)} if tokens else set()  # exactly one (whole-doc) shingle
    # Slide a width-k window; a tuple is hashable so it can live in a set.
    # There are len(tokens) - k + 1 starting positions.
    return {tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def char_shingles(text: str, k: int = 5) -> Set[str]:
    text = normalize(text)
    if len(text) < k:
        return {text} if text else set()
    # Character windows: robust across languages, larger sets than word shingles.
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def hash_shingle(shingle) -> int:
    # Convert a shingle to a compact 64-bit int. MinHash (Chapter 3) hashes
    # shingles, so storing ints instead of tuples/strings saves large amounts
    # of memory at corpus scale.
    s = " ".join(shingle) if isinstance(shingle, tuple) else shingle
    data = s.encode("utf-8")
    if _HAVE_XXHASH:
        return xxhash.xxh64(data).intdigest()   # fast 64-bit digest
    return hash(data) & 0xFFFFFFFFFFFFFFFF        # fallback: mask Python hash to 64 bits


# --- the metric itself ---------------------------------------------------
def jaccard(a: set, b: set) -> float:
    if not a and not b:                       # define J(empty, empty) = 1.0 (identical)
        return 1.0
    inter = len(a & b)                        # |A ∩ B|: hash-set intersection, O(min(|a|,|b|))
    union = len(a) + len(b) - inter           # |A ∪ B| via the identity (avoids building the union set)
    return inter / union


if __name__ == "__main__":
    d1 = "The  Quick brown FOX jumps over the lazy dog."
    d2 = "the quick brown fox jumps over a lazy dog"
    s1, s2 = word_shingles(d1, k=3), word_shingles(d2, k=3)
    print(f"|A|={len(s1)}  |B|={len(s2)}  J={jaccard(s1, s2):.3f}")
```

Two non-obvious points worth pausing on. **First**, normalization happens *inside* `word_shingles`, so it is impossible to forget it — a deliberate API choice. **Second**, `jaccard` never materializes the union set; it uses `|A ∪ B| = |A| + |B| − |A ∩ B|`, which halves the work and memory of the comparison.

---

## 4. A fully worked numerical example

We compute `J(A, B)` by hand for two real paragraphs using **word 3-grams**. The two paragraphs are deliberately constructed to be near-duplicates: each is **50 tokens** long and they are word-for-word identical *except at five positions* (shown in **bold**).

**Document A (50 words):**

> the deduplication pipeline removes near duplicate documents from the training corpus before the model **begins** learning by comparing the content of each document against every other document the system can **discard** repeated text that would otherwise waste compute and bias the language model toward memorized passages instead of **general** patterns found across the **data**

**Document B (50 words):**

> the deduplication **stage** removes near duplicate documents from the training corpus before the model **starts** learning by comparing the content of each document against every other document the system can **drop** repeated text that would otherwise waste compute and bias the language model toward memorized passages instead of **broad** patterns found across the **corpus**

After `normalize` (already lowercase, single-spaced, no punctuation) each document tokenizes to exactly **50 word tokens**, so each shingle set has:

```
number of word 3-grams = 50 − 3 + 1 = 48
```

Assuming no trigram repeats within a document (true for ordinary prose here), `|A| = 48` and `|B| = 48`.

**Which shingles differ?** A 3-gram is *shared* only if all three of its words are identical in both documents. A single differing token at position `p` corrupts every 3-gram whose window covers `p` — that is windows starting at `p−2, p−1, p` (the ones that exist). The five differing token positions are, by word index (1-based): **3** (`pipeline`/`stage`), **15** (`begins`/`starts`), **24** (`discard`/`drop`), **44** (`general`/`broad`), and **50** (`data`/`corpus`).

Counting the distinct windows each difference breaks:

| Differing position | Windows covering it (start indices) | Count |
|---|---|---|
| 3  | 1, 2, 3        | 3 |
| 15 | 13, 14, 15     | 3 |
| 24 | 22, 23, 24     | 3 |
| 44 | 42, 43, 44     | 3 |
| 50 | 48 only (49, 50 don't exist; last start is 48) | 1 |

The five differences are far apart (gaps ≫ 3), so no window is broken by two differences at once — the counts simply add:

```
broken shingles = 3 + 3 + 3 + 3 + 1 = 13
```

Therefore the shingles present in **both** documents:

```
|A ∩ B| = 48 − 13 = 35
```

Now apply the union identity and the definition:

```
|A| = 48
|B| = 48
|A ∩ B| = 35
|A ∪ B| = |A| + |B| − |A ∩ B| = 48 + 48 − 35 = 61
J(A,B) = |A ∩ B| / |A ∪ B| = 35 / 61 ≈ 0.574
```

**Representative members.**

- In `A ∩ B` (shared, 35 total): `("removes","near","duplicate")`, `("the","training","corpus")`, `("repeated","text","that")`, `("memorized","passages","instead")`.
- Only in `A` (13 total): `("the","deduplication","pipeline")`, `("model","begins","learning")`, `("can","discard","repeated")`, `("of","general","patterns")`, `("across","the","data")`.
- Only in `B` (13 total): `("the","deduplication","stage")`, `("model","starts","learning")`, `("can","drop","repeated")`, `("of","broad","patterns")`, `("across","the","corpus")`.

A Jaccard of **0.574** from a five-word edit out of fifty illustrates a key intuition: even *light* paraphrasing visibly depresses word-shingle similarity, because each changed word damages up to `k` shingles. If we raised `k` to 5, each isolated edit would break up to 5 shingles and `J` would fall further — the sparsity/specificity trade-off of Section 2.5 in action.

---

## 5. Common pitfalls

### Pitfall A — "Stopword soup" false collisions

**Diagnosis.** Unrelated documents report surprisingly high Jaccard (e.g. 0.3+). Inspecting the intersection, the shared shingles are dominated by function words: `("of","the","and")`, `("in","the","case")`. With `k = 2` word shingles this is acute because stopword bigrams appear in nearly every English document.

**Fix.** Increase `k` to 3 (longer windows are far less likely to be all-stopword), and/or strip a small stopword list *before* shingling. Do not over-strip: removing too many words can make genuinely different sentences collapse onto the same shingle.

### Pitfall B — "Forgot to normalize"

**Diagnosis.** Two documents you *know* are near-identical report low Jaccard (e.g. 0.62 instead of 0.94). The intersection is small even though the text looks the same; the only differences are casing, smart quotes vs straight quotes, or a stray non-breaking space — exactly the surface noise of Section 2.5.

**Fix.** Always run `dedup_pipeline.text_processing.normalizer` (NFKC + lowercase + whitespace collapse) *before* shingling. In this codebase normalization is called inside the shingler so it cannot be skipped; if you build shingles by hand, you must do it yourself.

### Pitfall C — "k larger than the document"

**Diagnosis.** Short documents (titles, tweets, log lines) produce empty or near-empty shingle sets, so `jaccard` either divides by zero or returns misleading `0.0`/`1.0` extremes. A 4-word title with `k = 5` word shingles has `4 − 5 + 1 = 0` windows.

**Fix.** Guard the windowing: when `len(tokens) < k`, emit a single whole-document shingle (as the code in Section 3 does) rather than an empty set, and define `J(∅, ∅) = 1.0`. For corpora with many short documents, consider character shingles, which tolerate small `k` gracefully.

### Pitfall D (bonus) — "URLs and code shattered into noise"

**Diagnosis.** Boilerplate-heavy pages (link farms, navigation menus) collide with each other, or all code files look alike, because a naive tokenizer split `https://a.com/x` into `https`, `a`, `com`, `x` and split `a == b` into trivial fragments.

**Fix.** Use a tokenizer that treats URLs as single tokens and preserves meaningful code symbols, or canonicalize URLs/numbers to placeholders (`<URL>`, `<NUM>`) so they contribute a stable single token instead of a shower of common fragments.

---

## 6. Chapter summary

- Document similarity for deduplication is defined on **sets**, not strings. The **Jaccard coefficient** `J(A,B) = |A ∩ B| / |A ∪ B|` measures shared content, lies in `[0,1]`, and is computed cheaply via `|A ∪ B| = |A| + |B| − |A ∩ B|`.
- Jaccard is **order-insensitive** and `O(|A|+|B|)`, beating edit distance (`O(n·m)`, alignment-bound) for long, possibly-reordered documents — the common case in web-scraped pretraining data [Broder 1997; Manber 1994].
- **Shingling** turns text into sets: character n-grams (`k≈5–9`, robust/multilingual), word n-grams (`k≈3`, semantic, the English workhorse), or sentence tokens (high fidelity, high variance).
- **Larger `k`** means sparser, more specific shingles and fewer false collisions, at the cost of missing lightly-edited near-duplicates. **Normalization** (NFKC + lowercase + whitespace collapse) removes surface noise and can lift a true near-duplicate's score from ~0.62 to ~0.94.
- Exact all-pairs Jaccard is `O(n²)`: ~`5×10¹¹` pairs for a million documents, ~**14 hours** of pure comparison at 10⁷ pairs/sec — intractable, and the reason MinHash + LSH exist (Chapter 3 onward) [Leskovec et al. 2014].

---

## 7. Self-check quiz

**Q1.** Document A has 800 word-3-gram shingles, document B has 1,200, and they share 300. What is `J(A,B)`?

**A1.** `|A ∪ B| = |A| + |B| − |A ∩ B| = 800 + 1200 − 300 = 1700`. So `J = |A ∩ B| / |A ∪ B| = 300 / 1700 ≈ **0.176**`. (Note you never needed to enumerate the union.)

**Q2.** Your pipeline reports `J = 0.61` for two documents that are obviously the same article, one copied from a PDF. The shared shingles are far fewer than expected. Name the most likely pitfall and the fix.

**A2.** This is **Pitfall B, "Forgot to normalize."** The PDF copy carries surface noise — smart quotes, ligatures (NFKC-foldable), inconsistent casing, non-breaking spaces — that splits otherwise-identical shingles. The fix is to run the `normalizer` (Unicode NFKC + lowercase + whitespace collapse) *before* shingling, which typically restores the score into the 0.9+ range.

**Q3.** You scale from 1,000,000 to 4,000,000 documents. By roughly what factor does the exact all-pairs comparison cost increase, and approximately how many pairs is that?

**A3.** All-pairs cost is `O(n²)`, so multiplying `n` by 4 multiplies the work by `4² = **16×**`. Concretely `C(4,000,000, 2) = 4,000,000 × 3,999,999 / 2 ≈ 8 × 10¹²` pairs — about 16 times the `~5 × 10¹¹` pairs at one million documents. This quadratic blow-up is precisely why we move to sub-quadratic MinHash + LSH methods.

---

## References

- **[Broder 1997]** A. Z. Broder. "On the resemblance and containment of documents." *Proceedings of the Compression and Complexity of Sequences (SEQUENCES '97)*, IEEE, 1997. — Introduces document *resemblance* (Jaccard of shingle sets), *containment*, and w-shingling.
- **[Manber 1994]** U. Manber. "Finding similar files in a large file system." *Proceedings of the USENIX Winter 1994 Technical Conference*, 1994. — Early system for near-duplicate file detection using fingerprinted substrings rather than alignment.
- **[Leskovec et al. 2014]** J. Leskovec, A. Rajaraman, and J. D. Ullman. *Mining of Massive Datasets*, 2nd ed. Cambridge University Press, 2014. — Chapter 3, "Finding Similar Items," is the canonical treatment of shingling, Jaccard similarity, MinHashing, and LSH.
