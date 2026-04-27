"""Candidate-match scoring tool — pure deterministic, no IO."""

from agents import function_tool

from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.source_lookup import SelectedCandidate

# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------

#: Jaccard token-overlap on item name vs candidate name.
W_NAME: float = 0.60

#: Bonus when the item brand appears in the candidate's source_name.
W_BRAND: float = 0.20

#: Bonus when the item category appears in the candidate's source_name.
W_CATEGORY: float = 0.10

#: Source-type reliability prior.
W_SOURCE: float = 0.10

#: Per-source reliability priors (USDA > OFF > web).
SOURCE_PRIORS: dict[str, float] = {
    "usda": 1.0,
    "off": 0.8,
    "web": 0.5,
}


def _recall(item_name: str, source_name: str) -> float:
    """Compute token recall: fraction of item's tokens found in source_name.

    Recall rewards candidates that contain all the item's key tokens even
    when the source name is longer (e.g. "Salmon" matches "Salmon Fillet"
    perfectly because every item token appears in the source).

    Args:
        item_name: The user's food item name (lowercased and split).
        source_name: The candidate's display name (lowercased and split).

    Returns:
        Recall in [0, 1]; 0.0 when *item_name* has no tokens.
    """
    item_tokens = set(item_name.lower().split())
    if not item_tokens:
        return 0.0
    source_tokens = set(source_name.lower().split())
    return len(item_tokens & source_tokens) / len(item_tokens)


def score_candidate_match(item: FoodItem, candidate: SelectedCandidate) -> float:
    """Score how well a candidate matches the user's food item.

    Combines four weighted signals into a scalar in [0, 1]:

    - **name** (60 %): Token recall — fraction of ``item.name`` tokens that
      appear in ``candidate.source_name`` (rewards candidates that contain all
      the item's key tokens even when the source name is longer).
    - **brand** (20 %): 1.0 if ``item.brand`` appears (case-insensitive) in
      ``candidate.source_name``, 0.0 otherwise.
    - **category** (10 %): 1.0 if ``item.category`` appears in
      ``candidate.source_name``, 0.0 otherwise.
    - **source prior** (10 %): USDA = 1.0, OFF = 0.8, web = 0.5 (reflects
      typical nutritional accuracy of each source type).

    Constants ``W_NAME``, ``W_BRAND``, ``W_CATEGORY``, ``W_SOURCE`` are
    module-level; ``SOURCE_PRIORS`` maps source tag to its prior.

    Args:
        item: The user's food item (provides ``name``, ``brand``, ``category``).
        candidate: A normalised candidate from any source.

    Returns:
        Match score clamped to [0.0, 1.0].
    """
    name_score = _recall(item.name, candidate.source_name)

    brand_score = 0.0
    if item.brand and candidate.source_name:
        if item.brand.lower() in candidate.source_name.lower():
            brand_score = 1.0

    category_score = 0.0
    if item.category and candidate.source_name:
        if item.category.lower() in candidate.source_name.lower():
            category_score = 1.0

    source_score = SOURCE_PRIORS.get(candidate.source, 0.5)

    total = (
        W_NAME * name_score
        + W_BRAND * brand_score
        + W_CATEGORY * category_score
        + W_SOURCE * source_score
    )

    return min(max(total, 0.0), 1.0)


# Pre-built FunctionTool for use in Agent(tools=[...]).
# Tests call `score_candidate_match(...)` directly; agent-domain imports this.
score_candidate_match_tool = function_tool(score_candidate_match)
