"""Select the best-matching candidate from a list — pure deterministic, no IO."""

from snaq_verify.application.tools.score_candidate_match import score_candidate_match
from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.source_lookup import SelectedCandidate


def select_best_candidate(
    item: FoodItem,
    candidates: list[SelectedCandidate],
    min_score: float,
) -> SelectedCandidate | None:
    """Return the highest-scoring candidate that meets *min_score*, or None.

    Scores every candidate with :func:`score_candidate_match`, then picks the
    best one.  Tie-break is deterministic: lowest ``source_id`` string
    (lexicographic ascending) wins, so the result is stable regardless of the
    input order.

    Args:
        item: The user's food item (provides ``name``, ``brand``, ``category``).
        candidates: Pool of normalised candidates from any source.
        min_score: Minimum acceptable score in [0, 1].  Candidates below this
            threshold are ignored.  Pass 0.0 to always pick the best.

    Returns:
        The best-matching :class:`SelectedCandidate`, or ``None`` if the pool
        is empty or no candidate reaches *min_score*.
    """
    if not candidates:
        return None

    scored: list[tuple[float, SelectedCandidate]] = [
        (score_candidate_match(item, c), c) for c in candidates
    ]

    # Sort: highest score first; for equal scores sort by source_id ascending.
    scored.sort(key=lambda t: (-t[0], t[1].source_id))

    best_score, best_candidate = scored[0]
    if best_score >= min_score:
        return best_candidate
    return None
