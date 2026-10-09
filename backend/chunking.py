"""Split a long document into overlapping pieces before embedding.

A single vector for a whole handbook mixes unrelated topics. A question
about annual leave then has to match a vector that also contains sick
leave, holidays, and payroll. Chunking gives each topic its own vector.

chunk_size is the window length in characters. chunk_overlap is how much
of the previous window is repeated in the next one, so a sentence that
sits on a boundary is not cut away from its neighbors. Character windows
are used here to keep the demo small. A production chunker usually counts
tokens and tries to break on paragraphs.
"""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Return sliding windows of `text`.

    Overlap must be smaller than the window, otherwise the window would
    never move forward.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    # Collapse whitespace left by the source formatting. This is the
    # small "clean" step in the indexing pipeline.
    cleaned = " ".join(text.split())
    if len(cleaned) <= chunk_size:
        return [cleaned]

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        window_end = min(start + chunk_size, len(cleaned))
        end = _end_on_boundary(cleaned, start, window_end)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(cleaned):
            break
        # Step back by the overlap, then land on a sentence or word so the
        # next chunk does not open in the middle of a word.
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        next_start = _start_on_sentence(cleaned, next_start, lookback=max(overlap, 80))
        if next_start <= start:
            next_start = _start_on_boundary(cleaned, start + 1)
        if next_start <= start or next_start >= len(cleaned):
            break
        start = next_start
    return chunks


def _end_on_boundary(text: str, start: int, end: int) -> int:
    """Stop the window on a sentence or a space when one is near the edge."""
    if end >= len(text):
        return len(text)
    window = text[start:end]
    sentence = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if sentence >= len(window) // 2:
        return start + sentence + 1
    space = window.rfind(" ")
    if space >= len(window) // 2:
        return start + space
    return end


def _start_on_boundary(text: str, start: int) -> int:
    """Move to the next word if `start` lands inside one."""
    if start <= 0:
        return 0
    if start < len(text) and text[start - 1] == " ":
        return start
    space = text.find(" ", start)
    if space == -1:
        return len(text)
    return space + 1


def _start_on_sentence(text: str, start: int, lookback: int) -> int:
    """Prefer the start of the sentence that contains `start`.

    Look back only a short distance. A chunk then begins on a sentence
    when one is nearby, and otherwise on a word.
    """
    start = _start_on_boundary(text, start)
    if start <= 0:
        return 0
    window_start = max(0, start - lookback)
    window = text[window_start:start]
    sentence = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
    if sentence >= 0:
        return window_start + sentence + 2
    return start
