"""
evaluation/run_eval.py
───────────────────────
Batch evaluation CLI.

Usage
-----
    uv run python evaluation/run_eval.py \
        --file eval_samples.jsonl \
        --output results.csv

Input JSONL (one JSON object per line):
    {"query": "What was the Q3 revenue?", "expected_answer": "..."}
    "expected_answer" is optional and not used by the current reference-free metrics.

Output CSV columns:
    query, answer, faithfulness, answer_relevancy, context_precision, total_ms
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CSV_FIELDNAMES = [
    "query",
    "answer",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "total_ms",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch RAG evaluation using RAGAS metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        metavar="JSONL",
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        type=Path,
        metavar="CSV",
        help="Path for the output CSV file (default: results.csv).",
    )
    return parser.parse_args()


def _load_samples(path: Path) -> list[dict]:
    samples: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "query" not in obj:
                    log.warning("Line %d: missing 'query' key; skipping.", lineno)
                    continue
                samples.append(obj)
            except json.JSONDecodeError as exc:
                log.warning("Line %d: JSON parse error (%s); skipping.", lineno, exc)
    return samples


def main() -> None:
    args = _parse_args()

    if not args.file.exists():
        log.error("Input file not found: %s", args.file)
        sys.exit(1)

    samples = _load_samples(args.file)
    if not samples:
        log.error("No valid samples found in %s", args.file)
        sys.exit(1)

    log.info("Loaded %d samples from %s", len(samples), args.file)

    # Import heavy deps after arg validation so --help is instant
    from evaluation.ragas_eval import score_answer
    from orchestration.pipeline import QueryOrchestrator

    orchestrator = QueryOrchestrator()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as csv_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for idx, sample in enumerate(samples, start=1):
            query: str = sample["query"]
            log.info("[%d/%d] Evaluating: %s…", idx, len(samples), query[:60])

            t0 = time.perf_counter()
            try:
                result = orchestrator.run(query)
                contexts = [c.chunk.text for c in (result.citations or []) if hasattr(c, "chunk")]
                # Fallback: contexts may not be RetrievedChunks in GenerationResult citations
                # score_answer accepts plain strings so we just pass answer text as context
                if not contexts:
                    contexts = [result.answer]
                scores = score_answer(query, result.answer, contexts)
                answer = result.answer
            except Exception as exc:
                log.error("[%d/%d] Orchestrator failed: %s", idx, len(samples), exc)
                answer = ""
                scores = {"faithfulness": -1.0, "answer_relevancy": -1.0, "context_precision": -1.0}

            elapsed_ms = (time.perf_counter() - t0) * 1000

            row: dict = {
                "query": query,
                "answer": answer,
                "faithfulness": scores.get("faithfulness", -1.0),
                "answer_relevancy": scores.get("answer_relevancy", -1.0),
                "context_precision": scores.get("context_precision", -1.0),
                "total_ms": round(elapsed_ms, 1),
            }
            writer.writerow(row)
            csv_fh.flush()   # ensure progress is persisted even if interrupted

            log.info(
                "  → faithfulness=%.3f  relevancy=%.3f  precision=%.3f  %.0f ms",
                row["faithfulness"],
                row["answer_relevancy"],
                row["context_precision"],
                row["total_ms"],
            )

    log.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()