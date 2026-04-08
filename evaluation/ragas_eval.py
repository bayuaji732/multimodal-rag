"""
evaluation/ragas_eval.py
─────────────────────────
RAGAS-based answer quality scoring without requiring ground-truth labels.

Metrics used
────────────
• faithfulness       — is the answer grounded in the retrieved context?
• answer_relevancy   — does the answer actually address the question?
• context_precision  — are the retrieved chunks relevant to the question?

All metrics operate without a reference/expected answer, making them safe for
live production use.

Usage
-----
    from evaluation.ragas_eval import score_answer
    scores = score_answer(query, answer, [chunk.text for chunk in chunks])
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

_FAILURE_SCORES: dict[str, float] = {
    "faithfulness": -1.0,
    "answer_relevancy": -1.0,
    "context_precision": -1.0,
}


def score_answer(
    query: str,
    answer: str,
    contexts: list[str],
) -> dict[str, float]:
    """
    Score an answer using RAGAS faithfulness, answer relevancy, and context precision.

    Parameters
    ----------
    query    : The original user question.
    answer   : The generated answer string.
    contexts : List of retrieved chunk texts used to generate the answer.

    Returns
    -------
    dict with keys {"faithfulness", "answer_relevancy", "context_precision"}.
    All values are floats in [0, 1].  On any error, returns all values as -1.0
    and logs the exception — never raises.
    """
    if not contexts:
        log.warning("score_answer: empty contexts list; returning failure scores.")
        return dict(_FAILURE_SCORES)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            faithfulness,
        )

        ds = Dataset.from_dict({
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
        })

        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        scores: dict[str, float] = {k: float(v) for k, v in result.items()}

        # Normalise to [0, 1] — RAGAS occasionally returns slightly out-of-range values
        scores = {k: max(0.0, min(1.0, v)) for k, v in scores.items()}

        log.info(
            "RAGAS scores — faithfulness=%.3f, answer_relevancy=%.3f, context_precision=%.3f",
            scores.get("faithfulness", -1),
            scores.get("answer_relevancy", -1),
            scores.get("context_precision", -1),
        )
        return scores

    except Exception as exc:
        log.error("score_answer failed: %s", exc, exc_info=True)
        return dict(_FAILURE_SCORES)