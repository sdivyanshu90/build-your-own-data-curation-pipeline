# %% [markdown]
# # Deduplicating AG News with `dedup_pipeline`
#
# This script is a Jupyter notebook in `# %%`-cell format (open it with
# Jupytext, VS Code, or PyCharm "Run Cell"). It:
#
# 1. Loads 10,000 documents from the AG News dataset via HuggingFace.
# 2. Inspects which documents would be flagged as duplicates *before* removal.
# 3. Runs the full deduplication pipeline.
# 4. Prints a statistics summary table.
# 5. Plots the cluster-size distribution histogram.
#
# Extra dependencies for this example only:
#
# ```bash
# pip install -e ".[hf]" matplotlib
# ```

# %%
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp

from datasets import load_dataset

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.evaluation.metrics import pairs_from_clusters
from dedup_pipeline.minhash.minhash import MinHasher
from dedup_pipeline.pipeline.pipeline import DedupPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# %% [markdown]
# ## 1. Load 10,000 AG News documents
#
# AG News stores the article text under the `"text"` field, which is the
# pipeline's default `text_field`. We materialise the first 10,000 rows as
# in-memory records `{"id", "text"}`.

# %%
N_DOCS = 10_000
dataset = load_dataset("ag_news", split=f"train[:{N_DOCS}]")
records: list[dict[str, str]] = [
    {"id": str(i), "text": row["text"]} for i, row in enumerate(dataset)
]
print(f"Loaded {len(records):,} AG News documents")
print("Example:", records[0]["text"][:120], "...")

# %% [markdown]
# ## 2. Configure the pipeline
#
# Character 5-grams are robust for short news text; 128 hash functions in
# 16 bands of 8 rows place the LSH S-curve knee near Jaccard 0.69, and we accept
# pairs at estimated Jaccard >= 0.8. `high_precision_mode` re-verifies every
# candidate pair against the threshold.

# %%
config = PipelineConfig(
    shingle_mode="char",
    shingle_size=5,
    num_hash_functions=128,
    lsh_bands=16,
    lsh_rows=8,
    jaccard_threshold=0.8,
    high_precision_mode=True,
    representative_strategy="longest",
    random_seed=42,
)
pipeline = DedupPipeline(config)

# %% [markdown]
# ## 3. Inspect flagged duplicate pairs BEFORE removal
#
# `detect_clusters` runs stages 1-8 (no output is written), returning the
# duplicate clusters. We expand them into pairs and print a few side by side with
# their MinHash-estimated Jaccard so a human can sanity-check before deleting.

# %%
clusters, n_docs = pipeline.detect_clusters(records)
predicted_pairs = sorted(pairs_from_clusters(clusters))
print(f"{n_docs:,} documents -> {len(clusters):,} duplicate clusters, "
      f"{len(predicted_pairs):,} duplicate pairs")

# Recompute signatures once to show the estimated Jaccard of sample pairs.
normalized = pipeline.normalize_batch(list(pipeline.stream_documents(records)))
signatures = pipeline.compute_signatures(pipeline.shingle_batch(normalized))

print("\n--- Sample flagged pairs (inspect before removal) ---")
for i, j in predicted_pairs[:5]:
    est = MinHasher.estimate_jaccard(signatures[i], signatures[j])
    print(f"\n[pair ({i}, {j})  estimated Jaccard = {est:.3f}]")
    print(f"  A: {records[i]['text'][:100]}")
    print(f"  B: {records[j]['text'][:100]}")

# %% [markdown]
# ## 4. Run the full pipeline and print a statistics summary table

# %%
out_dir = Path(mkdtemp())
dest = out_dir / "agnews_dedup.jsonl"
stats = pipeline.run(records, dest)

print("\n=========== Deduplication summary ===========")
rows = [
    ("Input documents", f"{stats['input_count']:,}"),
    ("Output documents", f"{stats['output_count']:,}"),
    ("Removed", f"{stats['input_count'] - stats['output_count']:,}"),
    ("Dedup rate", f"{stats['dedup_rate']:.2%}"),
    ("Duplicate clusters", f"{len(clusters):,}"),
    ("Output file", str(dest)),
]
width = max(len(label) for label, _ in rows)
for label, value in rows:
    print(f"  {label:<{width}} : {value}")
print("=============================================")

print("\nRuntime per stage (seconds):")
for stage, seconds in stats["runtime_per_stage"].items():
    print(f"  {stage:<28} {seconds:7.3f}")

# %% [markdown]
# ## 5. Cluster-size distribution histogram
#
# Duplicate clusters are heavy-tailed: most are small (size 2-3), with a few
# large boilerplate clusters. We plot the distribution on a log y-axis.

# %%
try:
    import matplotlib.pyplot as plt

    sizes = [len(c) for c in clusters]
    size_counts = Counter(sizes)
    xs = sorted(size_counts)
    ys = [size_counts[s] for s in xs]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(xs, ys, color="#3b6fb0")
    ax.set_yscale("log")
    ax.set_xlabel("Cluster size (number of near-duplicate documents)")
    ax.set_ylabel("Number of clusters (log scale)")
    ax.set_title("AG News duplicate cluster-size distribution")
    fig.tight_layout()
    hist_path = out_dir / "cluster_size_histogram.png"
    fig.savefig(hist_path, dpi=120)
    print(f"Saved histogram to {hist_path}")
    plt.show()
except ImportError:
    print("matplotlib not installed; skipping the histogram. "
          "Install it with: pip install matplotlib")

# %% [markdown]
# ## Done
#
# The cleaned corpus is at `dest` and a machine-readable statistics file sits
# next to it (`agnews_dedup_stats.json`). Swap `records` for a path/glob/dataset
# id to scale this up — the pipeline streams the source and never loads the whole
# corpus into memory at once.
