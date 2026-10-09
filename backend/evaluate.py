"""Run the evaluation set through the existing RAG pipeline.

Retrieval metrics use the top 5 chunks. The same five chunks are the
context for the generated answer, so answer accuracy and retrieval
quality describe one configuration. The run is repeated for the three
chunk presets. Unanswerable questions are scored as abstention and are
left out of precision, recall, and MRR.
"""

from __future__ import annotations

import re

from corpus import DOCUMENTS
from eval_set import EVAL_QUESTIONS, EvalQuestion
from metrics import mean, parent_document_id, precision_at_k, recall_at_k, reciprocal_rank
from pipelines import REFUSAL, KnowledgeBase
from vector_store import Hit

# Top 5 is what Precision@5 and Recall@5 both need.
EVAL_TOP_K = 5

# The chunking experiment. Names match the presets on the question screen.
PRESETS: list[dict] = [
    {"name": "A", "chunk_size": 200, "chunk_overlap": 20},
    {"name": "B", "chunk_size": 500, "chunk_overlap": 50},
    {"name": "C", "chunk_size": 1000, "chunk_overlap": 100},
]

TITLES = {document["id"]: document["title"] for document in DOCUMENTS}

# Written after the label is known. The question and the retrieved titles
# are filled in so the note describes this run, not a generic example.
FAILURE_NOTES = {
    "retrieval": (
        "Retrieval",
        "The labeled document did not appear in the top 5.",
        "The nearest chunks came from other documents, so the answer was built without the labeled source.",
        "Try another chunk size, or add keyword search for exact figures and names.",
    ),
    "chunking": (
        "Chunking",
        "The labeled document was retrieved, but the expected fact was not inside those chunks.",
        "The fact sits in a part of the document that this window size did not keep with the matching passage.",
        "Change chunk size and overlap so the fact and its surrounding sentence stay in one chunk.",
    ),
    "generation": (
        "Generation",
        "The expected fact was available in the retrieved chunks, but the answer did not state it.",
        "The model or the extractive fallback used the wrong passage, or it answered when it should have refused.",
        "Tighten the prompt, lower the similarity bar for refusals, or pass fewer noisy chunks.",
    ),
    "citation": (
        "Citation",
        "The answer contained the expected fact, but it did not name the labeled source.",
        "The fact was produced without pointing at the document the label names.",
        "Require the source title in the answer, and check it against the retrieved chunk.",
    ),
}


def evaluate_all(base: KnowledgeBase) -> dict:
    """Score every preset. Preset B supplies the failure write-ups first."""
    presets = [evaluate_preset(base, preset) for preset in PRESETS]
    return {
        "question_count": len(EVAL_QUESTIONS),
        "presets": presets,
        "failures": _failure_writeups(presets),
    }


def evaluate_preset(base: KnowledgeBase, preset: dict) -> dict:
    rows = [
        _score_question(
            base,
            question,
            preset["chunk_size"],
            preset["chunk_overlap"],
        )
        for question in EVAL_QUESTIONS
    ]
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    return {
        "name": preset["name"],
        "chunk_size": preset["chunk_size"],
        "chunk_overlap": preset["chunk_overlap"],
        "precision_at_3": _round(mean([row["precision_at_3"] for row in answerable])),
        "precision_at_5": _round(mean([row["precision_at_5"] for row in answerable])),
        "recall_at_5": _round(mean([row["recall_at_5"] for row in answerable])),
        "mrr": _round(mean([row["reciprocal_rank"] for row in answerable])),
        "answer_accuracy": _round(mean([1.0 if row["correct"] else 0.0 for row in answerable])),
        "abstention_accuracy": _round(
            mean([1.0 if row["correct"] else 0.0 for row in unanswerable])
        ),
        "rows": rows,
    }


def _score_question(
    base: KnowledgeBase,
    question: EvalQuestion,
    chunk_size: int,
    overlap: int,
) -> dict:
    result = base.answer(question.question, EVAL_TOP_K, chunk_size, overlap)
    document_ids = [parent_document_id(hit.record.id) for hit in result.hits]
    gold = set(question.document_ids)
    gold_text = _gold_text(result.hits, gold)
    answer = result.answer
    correct = _is_correct(question, answer, result.refused)
    grounded = _is_grounded(question, answer, result.refused, gold_text, result.hits)
    citation = _citation_correct(question, answer, result.refused)
    failure = _failure_type(question, document_ids, gold, gold_text, correct, citation)
    retrieved = [
        {
            "rank": hit.rank,
            "title": hit.record.title,
            "score": round(hit.score, 4),
            "document_id": document_id,
        }
        for hit, document_id in zip(result.hits, document_ids)
    ]
    return {
        "id": question.id,
        "question": question.question,
        "type": question.type,
        "expected_answer": question.expected_answer,
        "source": question.source,
        "page": question.page,
        "answerable": question.answerable,
        "answer": answer,
        "refused": result.refused,
        "generation_mode": result.generation_mode,
        "retrieved": retrieved,
        "precision_at_3": _round(precision_at_k(document_ids, gold, 3)) if gold else None,
        "precision_at_5": _round(precision_at_k(document_ids, gold, 5)) if gold else None,
        "recall_at_5": _round(recall_at_k(document_ids, gold, 5)) if gold else None,
        "reciprocal_rank": _round(reciprocal_rank(document_ids, gold)) if gold else None,
        "correct": correct,
        "grounded": grounded,
        "citation_correct": citation,
        "failure": failure,
    }


def _is_correct(question: EvalQuestion, answer: str, refused: bool) -> bool:
    if not question.answerable:
        # Abstention is correct. A nearest-neighbor quote is not.
        return refused or _contains(answer, REFUSAL)
    return _contains(answer, question.expected_answer)


def _is_grounded(
    question: EvalQuestion,
    answer: str,
    refused: bool,
    gold_text: str,
    hits: list[Hit],
) -> bool:
    """True when the reply does not state a fact the retrieved gold text lacks.

    A refusal is grounded: it adds nothing. An answer that states the
    expected fact is grounded only when that fact is in a retrieved gold
    chunk. Any other reply is grounded when it is copied from the chunks
    that were actually retrieved.
    """
    if not question.answerable:
        # Refusing, including in the answer text, adds no unsupported fact.
        return refused or _contains(answer, REFUSAL)
    if refused:
        return True
    if _contains(answer, question.expected_answer):
        return _contains(gold_text, question.expected_answer)
    return _copied_from_hits(answer, hits)


def _citation_correct(question: EvalQuestion, answer: str, refused: bool) -> bool:
    """The reply names a labeled source. A refusal has nothing to cite."""
    if refused or not question.answerable:
        return False
    folded = answer.lower()
    return any(TITLES[document_id].lower() in folded for document_id in question.document_ids)


def _failure_type(
    question: EvalQuestion,
    document_ids: list[str],
    gold: set[str],
    gold_text: str,
    correct: bool,
    citation: bool,
) -> str:
    """Separate retrieval, chunking, generation, and citation.

    Order matters. A missing document is a retrieval failure even if the
    answer is also wrong. Chunking is only used when the document was
    found but the fact was not in the retrieved pieces.
    """
    if not question.answerable:
        return "none" if correct else "generation"
    if gold - set(document_ids):
        return "retrieval"
    if not _contains(gold_text, question.expected_answer):
        return "chunking"
    if not correct:
        return "generation"
    if not citation:
        return "citation"
    return "none"


def _failure_writeups(presets: list[dict]) -> list[dict]:
    """Up to five failures, preset B first, worst reciprocal rank first."""
    ordered = sorted(presets, key=lambda preset: 0 if preset["name"] == "B" else 1)
    chosen: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for preset in ordered:
        rows = [row for row in preset["rows"] if row["failure"] != "none"]
        rows.sort(key=lambda row: row["reciprocal_rank"] if row["reciprocal_rank"] is not None else -1)
        for row in rows:
            key = (preset["name"], row["id"])
            if key in seen:
                continue
            seen.add(key)
            chosen.append(_writeup(preset, row))
            if len(chosen) == 5:
                return chosen
    return chosen


def _writeup(preset: dict, row: dict) -> dict:
    stage, what, why, fix = FAILURE_NOTES[row["failure"]]
    titles = ", ".join(item["title"] for item in row["retrieved"]) or "(none)"
    return {
        "preset": preset["name"],
        "id": row["id"],
        "question": row["question"],
        "failure": row["failure"],
        "stage": stage,
        "what": what,
        "why": f"{why} Retrieved: {titles}.",
        "fix": fix,
    }


def _gold_text(hits: list[Hit], gold: set[str]) -> str:
    parts = [
        hit.record.text
        for hit in hits
        if parent_document_id(hit.record.id) in gold
    ]
    return "\n".join(parts)


def _contains(haystack: str, expected: str) -> bool:
    """True when every phrase in `expected` appears as its own words.

    A plain substring would treat "6 weeks" as present inside "16 weeks".
    The boundary is applied only at ends that are letters or digits, so a
    figure such as "17.4%" still matches.
    """
    folded = haystack.lower()
    for part in expected.split("|"):
        phrase = part.strip().lower()
        if not phrase:
            continue
        body = re.escape(phrase)
        prefix = r"\b" if phrase[0].isalnum() else ""
        suffix = r"\b" if phrase[-1].isalnum() else ""
        if re.search(prefix + body + suffix, folded) is None:
            return False
    return True


def _copied_from_hits(answer: str, hits: list[Hit]) -> bool:
    for hit in hits:
        sample = hit.record.text[:80]
        if len(sample) >= 40 and sample in answer:
            return True
    return False


def _round(value: float) -> float:
    return round(value, 4)
