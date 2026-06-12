"""Typer command-line interface for the deduplication pipeline.

Commands:
    * ``deduplicate`` — run the full pipeline and write a cleaned corpus.
    * ``evaluate`` — score predictions against a ground-truth pair file.
    * ``tune`` — grid-search ``(k, b, r, threshold)`` against ground truth.
    * ``benchmark`` — measure documents/second at a given corpus size.
    * ``inspect`` — print corpus statistics without deduplicating.

Responsibility:
    * Parse arguments, load configuration, and dispatch to the pipeline/eval.

Inputs:
    * CLI arguments (sources, destinations, config files).

Outputs:
    * Console reports and, where requested, output/result files.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Annotated, Any, Optional

import typer

from dedup_pipeline.config import PipelineConfig
from dedup_pipeline.evaluation.metrics import pairs_from_clusters, precision_recall_f1
from dedup_pipeline.exceptions import DedupError, EvaluationError
from dedup_pipeline.pipeline.pipeline import DedupPipeline
from dedup_pipeline.pipeline.reader import DocumentReader

logger = logging.getLogger("dedup_pipeline")

app = typer.Typer(
    add_completion=False,
    help="MinHash/LSH near-duplicate detection and removal for text corpora.",
)

# Grid values searched by the `tune` command (kept small for tractability).
_TUNE_SHINGLE_SIZES: tuple[int, ...] = (3, 5)
_TUNE_THRESHOLDS: tuple[float, ...] = (0.7, 0.8, 0.9)


def _configure_logging(verbose: bool) -> None:
    """Configure root logging for the CLI.

    Args:
        verbose: If ``True``, log at DEBUG; otherwise INFO.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _load_config(config_path: Path | None, **overrides: Any) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from a JSON file, with overrides.

    Args:
        config_path: Optional path to a JSON config file.
        **overrides: Field overrides applied on top of the file/defaults.

    Returns:
        A validated configuration.

    Raises:
        typer.BadParameter: If the file is missing or invalid JSON.
    """
    data: dict[str, Any] = {}
    if config_path is not None:
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise typer.BadParameter(
                f"could not read config {config_path}: {exc}"
            ) from exc
    data.update({k: v for k, v in overrides.items() if v is not None})
    try:
        return PipelineConfig(**data)
    except DedupError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _load_ground_truth(path: Path) -> set[tuple[int, int]]:
    """Load ground-truth duplicate pairs from a JSON file.

    The file may contain either a bare list of ``[i, j]`` pairs or an object
    with a ``"pairs"`` key holding that list.

    Args:
        path: The ground-truth JSON file.

    Returns:
        A set of canonical ``(i, j)`` pairs with ``i < j``.

    Raises:
        typer.BadParameter: If the file is missing or malformed.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(
            f"could not read ground truth {path}: {exc}"
        ) from exc
    pairs_raw = raw["pairs"] if isinstance(raw, dict) else raw
    try:
        return {
            (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
            for a, b in pairs_raw
        }
    except (TypeError, ValueError) as exc:
        raise typer.BadParameter(
            f"malformed ground-truth pairs in {path}: {exc}"
        ) from exc


@app.command()
def deduplicate(
    source: Annotated[str, typer.Option(help="Input path/glob/dataset id.")],
    dest: Annotated[Path, typer.Option(help="Output corpus path.")],
    config: Annotated[Optional[Path], typer.Option(help="JSON config file.")] = None,
    resume: Annotated[
        bool, typer.Option(help="Resume from checkpoints if present.")
    ] = False,
    verbose: Annotated[bool, typer.Option(help="DEBUG logging.")] = False,
) -> None:
    """Run the full pipeline and write a deduplicated corpus."""
    _configure_logging(verbose)
    cfg = _load_config(config)
    pipeline = DedupPipeline(cfg)
    try:
        stats = pipeline.run(source, dest, resume=resume)
    except DedupError as exc:
        typer.secho(f"Deduplication failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Done: {stats['input_count']} -> {stats['output_count']} docs "
        f"(dedup_rate={stats['dedup_rate']:.4f}); stats at "
        f"{pipeline._writer.stats_path_for(dest)}"
    )


@app.command()
def evaluate(
    source: Annotated[str, typer.Option(help="Input path/glob/dataset id.")],
    ground_truth: Annotated[
        Path, typer.Option("--ground-truth", help="JSON ground-truth pairs.")
    ],
    config: Annotated[Optional[Path], typer.Option(help="JSON config file.")] = None,
    verbose: Annotated[bool, typer.Option(help="DEBUG logging.")] = False,
) -> None:
    """Score predicted duplicates against a ground-truth pair file."""
    _configure_logging(verbose)
    cfg = _load_config(config)
    truth = _load_ground_truth(ground_truth)
    pipeline = DedupPipeline(cfg)
    try:
        clusters, n_docs = pipeline.detect_clusters(source)
    except DedupError as exc:
        typer.secho(f"Evaluation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    predicted = pairs_from_clusters(clusters)
    precision, recall, f1 = precision_recall_f1(predicted, truth)
    typer.echo(f"Documents:        {n_docs}")
    typer.echo(f"Predicted pairs:  {len(predicted)}")
    typer.echo(f"Ground-truth:     {len(truth)}")
    typer.echo(f"Precision:        {precision:.4f}")
    typer.echo(f"Recall:           {recall:.4f}")
    typer.echo(f"F1:               {f1:.4f}")


@app.command()
def tune(
    source: Annotated[str, typer.Option(help="Input path/glob/dataset id.")],
    ground_truth: Annotated[
        Path, typer.Option("--ground-truth", help="JSON ground-truth pairs.")
    ],
    output: Annotated[Path, typer.Option(help="Where to write tuning results JSON.")],
    config: Annotated[Optional[Path], typer.Option(help="Base JSON config file.")] = None,
    verbose: Annotated[bool, typer.Option(help="DEBUG logging.")] = False,
) -> None:
    """Grid-search (k, b, r, threshold) and write a results/Pareto JSON."""
    _configure_logging(verbose)
    base = _load_config(config)
    truth = _load_ground_truth(ground_truth)
    num_hashes = base.num_hash_functions
    # Factorizations of the fixed signature length give the (b, r) grid.
    factorizations = [
        (b, num_hashes // b) for b in range(1, num_hashes + 1) if num_hashes % b == 0
    ]
    results: list[dict[str, Any]] = []
    for shingle_size in _TUNE_SHINGLE_SIZES:
        for bands, rows in factorizations:
            for threshold in _TUNE_THRESHOLDS:
                try:
                    cfg = base.model_copy(
                        update={
                            "shingle_size": shingle_size,
                            "lsh_bands": bands,
                            "lsh_rows": rows,
                            "jaccard_threshold": threshold,
                        }
                    )
                    # model_copy bypasses validators; re-validate explicitly.
                    cfg = PipelineConfig(**cfg.model_dump())
                except DedupError:
                    continue  # invalid (b, r) for this n -> skip silently
                clusters, _ = DedupPipeline(cfg).detect_clusters(source)
                precision, recall, f1 = precision_recall_f1(
                    pairs_from_clusters(clusters), truth
                )
                results.append(
                    {
                        "shingle_size": shingle_size,
                        "lsh_bands": bands,
                        "lsh_rows": rows,
                        "jaccard_threshold": threshold,
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                    }
                )
    if not results:
        raise typer.BadParameter("no valid configurations were evaluated")
    results.sort(key=lambda r: r["f1"], reverse=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    best = results[0]
    typer.echo(f"Evaluated {len(results)} configs; results -> {output}")
    typer.echo(
        "Best F1={f1:.4f} at k={shingle_size}, b={lsh_bands}, r={lsh_rows}, "
        "threshold={jaccard_threshold}".format(**best)
    )


@app.command()
def benchmark(
    source: Annotated[str, typer.Option(help="Input path/glob/dataset id.")],
    n_docs: Annotated[int, typer.Option("--n-docs", help="Documents to process.")],
    config: Annotated[Optional[Path], typer.Option(help="JSON config file.")] = None,
    verbose: Annotated[bool, typer.Option(help="DEBUG logging.")] = False,
) -> None:
    """Measure documents/second over the first ``n_docs`` documents."""
    _configure_logging(verbose)
    cfg = _load_config(config)
    reader = DocumentReader(cfg)
    records: list[dict[str, Any]] = []
    for doc in reader.stream(source):
        records.append({cfg.id_field: doc.id, cfg.text_field: doc.text, **doc.metadata})
        if len(records) >= n_docs:
            break
    if not records:
        raise typer.BadParameter("source yielded no documents")
    pipeline = DedupPipeline(cfg)
    start = time.perf_counter()
    clusters, count = pipeline.detect_clusters(records)
    elapsed = time.perf_counter() - start
    throughput = count / elapsed if elapsed > 0 else float("inf")
    typer.echo(f"Documents:   {count}")
    typer.echo(f"Clusters:    {len(clusters)}")
    typer.echo(f"Elapsed:     {elapsed:.3f}s")
    typer.echo(f"Throughput:  {throughput:,.0f} docs/sec")


@app.command()
def inspect(
    source: Annotated[str, typer.Option(help="Input path/glob/dataset id.")],
    config: Annotated[Optional[Path], typer.Option(help="JSON config file.")] = None,
    verbose: Annotated[bool, typer.Option(help="DEBUG logging.")] = False,
) -> None:
    """Print corpus statistics without deduplicating."""
    _configure_logging(verbose)
    cfg = _load_config(config)
    reader = DocumentReader(cfg)
    count = 0
    total_chars = 0
    min_chars: int | None = None
    max_chars = 0
    exact_hashes: set[int] = set()
    exact_dupes = 0
    for doc in reader.stream(source):
        count += 1
        length = len(doc.text)
        total_chars += length
        min_chars = length if min_chars is None else min(min_chars, length)
        max_chars = max(max_chars, length)
        # Cheap exact-duplicate estimate via content hashing.
        digest = hash(doc.text)
        if digest in exact_hashes:
            exact_dupes += 1
        else:
            exact_hashes.add(digest)
    if count == 0:
        raise EvaluationError("source yielded no documents")
    typer.echo(f"Documents:           {count}")
    typer.echo(f"Avg length (chars):  {total_chars / count:,.1f}")
    typer.echo(f"Min/Max length:      {min_chars}/{max_chars}")
    typer.echo(f"Exact duplicates:    {exact_dupes} ({exact_dupes / count:.2%})")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
