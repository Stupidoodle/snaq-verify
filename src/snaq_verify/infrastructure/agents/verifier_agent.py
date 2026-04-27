"""Verifier agent factory — builds the OpenAI Agents SDK Agent for food verification."""

from __future__ import annotations

from dataclasses import dataclass

from agents import Agent, RunContextWrapper, function_tool

from snaq_verify.application.tools.check_atwater_consistency import (
    check_atwater_consistency_tool,
)
from snaq_verify.application.tools.compute_per_nutrient_delta import (
    compute_per_nutrient_delta_tool,
)
from snaq_verify.application.tools.format_human_summary import (
    format_human_summary_tool,
)
from snaq_verify.application.tools.select_best_candidate import (
    select_best_candidate_tool,
)
from snaq_verify.application.tools.verdict_from_deltas import verdict_from_deltas_tool
from snaq_verify.core.config import Settings
from snaq_verify.domain.models.enums import USDADataType
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.models.source_lookup import (
    OFFProduct,
    USDACandidate,
    WebSnippet,
)
from snaq_verify.domain.ports.open_food_facts_client_port import OpenFoodFactsClientPort
from snaq_verify.domain.ports.tavily_client_port import TavilyClientPort
from snaq_verify.domain.ports.usda_client_port import USDAClientPort
from snaq_verify.infrastructure.agents.guardrails.atwater_output_guardrail import (
    atwater_output_guardrail,
)
from snaq_verify.infrastructure.agents.guardrails.schema_output_guardrail import (
    schema_output_guardrail,
)

# ---------------------------------------------------------------------------
# Context injected into every tool call via RunContextWrapper
# ---------------------------------------------------------------------------


@dataclass
class VerifierContext:
    """Holds the IO ports for the verifier agent's tool closures.

    Passed to ``Runner.run(..., context=VerifierContext(...))`` so that
    each tool can reach the real adapters without importing them globally
    (which would break test isolation).
    """

    usda: USDAClientPort
    off: OpenFoodFactsClientPort
    tavily: TavilyClientPort
    settings: Settings


# ---------------------------------------------------------------------------
# IO tools — async, context-injected
# ---------------------------------------------------------------------------


@function_tool
async def search_usda(
    ctx: RunContextWrapper[VerifierContext],
    query: str,
    data_type: str = "Foundation",
) -> list[USDACandidate]:
    """Search USDA FoodData Central for matching food candidates.

    Try "Foundation" first for generic whole foods; fall back to "SR Legacy"
    if fewer than 2 results are returned.

    Args:
        ctx: Run context carrying the USDA client.
        query: Food name to search (e.g. "Chicken Breast Raw").
        data_type: USDA data-type filter — one of "Foundation", "SR Legacy",
            "Branded", or "Survey (FNDDS)".  Defaults to "Foundation".

    Returns:
        List of ``USDACandidate`` objects ordered by USDA relevance score.
        ``nutrition_per_100g`` is *not* populated at this stage — call
        ``get_usda_food`` to hydrate a candidate.
    """
    try:
        dt: USDADataType | None = USDADataType(data_type)
    except ValueError:
        dt = None
    return await ctx.context.usda.search(query, data_type=dt)


@function_tool
async def get_usda_food(
    ctx: RunContextWrapper[VerifierContext],
    fdc_id: int,
) -> USDACandidate:
    """Fetch the full nutrition payload for a specific USDA FDC entry.

    Args:
        ctx: Run context carrying the USDA client.
        fdc_id: Numeric FoodData Central food ID (e.g. 2646170).

    Returns:
        ``USDACandidate`` with ``nutrition_per_100g`` fully populated.
    """
    return await ctx.context.usda.get_food(fdc_id)


@function_tool
async def lookup_off_by_barcode(
    ctx: RunContextWrapper[VerifierContext],
    barcode: str,
) -> OFFProduct | None:
    """Look up a product in Open Food Facts by its barcode.

    Args:
        ctx: Run context carrying the OFF client.
        barcode: EAN or UPC barcode string (e.g. "5200435000027").

    Returns:
        The matching ``OFFProduct``, or ``None`` when OFF returns 404.
        A ``None`` return is normal for region-specific products — fall back
        to ``search_off_by_name``.
    """
    return await ctx.context.off.lookup_by_barcode(barcode)


@function_tool
async def search_off_by_name(
    ctx: RunContextWrapper[VerifierContext],
    name: str,
    brand: str | None = None,
) -> list[OFFProduct]:
    """Search Open Food Facts by product name, optionally filtered by brand.

    Args:
        ctx: Run context carrying the OFF client.
        name: Product name (e.g. "Total 0% Greek Yogurt").
        brand: Optional brand filter (e.g. "Fage").

    Returns:
        List of matching ``OFFProduct`` objects ordered by OFF relevance.
    """
    return await ctx.context.off.search_by_name(name, brand=brand)


@function_tool
async def search_tavily(
    ctx: RunContextWrapper[VerifierContext],
    query: str,
) -> list[WebSnippet]:
    """Search the web for nutrition information via Tavily.

    Use as a fallback when USDA and OFF together return fewer than 2 usable
    candidates with nutrition data.

    Args:
        ctx: Run context carrying the Tavily client.
        query: Search query (e.g. "Avocado Raw nutrition per 100g USDA").

    Returns:
        Ranked ``WebSnippet`` objects with ``title``, ``url``, and ``content``.
    """
    return await ctx.context.tavily.search(query)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_VERIFIER_INSTRUCTIONS = """\
You are a precision nutrition-data verification specialist. Your task is to verify
the reported nutrition values for a single food item against authoritative external
sources, then return a structured ItemVerification.

IMPORTANT: Never fabricate numeric values. Every number in the output must trace to
a tool result.

Follow these steps IN ORDER:

1. USDA LOOKUP
   - Call search_usda(query=item.name, data_type="Foundation").
   - If fewer than 2 results: also call search_usda(..., data_type="SR Legacy").
   - For each promising candidate (top 3 max), call get_usda_food(fdc_id=...) to
     retrieve full nutrition_per_100g.

2. OPEN FOOD FACTS LOOKUP
   - If the item has a barcode, call lookup_off_by_barcode(barcode=item.barcode).
   - Also call search_off_by_name(name=item.name, brand=item.brand).
   - Use only OFFProducts where nutrition_per_100g is populated.

3. WEB FALLBACK
   - If USDA + OFF together yield fewer than 2 hydrated candidates, call
     search_tavily(query="{item.name} nutrition per 100g").

4. CANDIDATE SELECTION (per source)
   - Convert each hydrated source result into a SelectedCandidate:
       source: "usda" | "off" | "web"
       source_id: str(fdc_id) | barcode | url
       source_name: description | product_name | title
       nutrition_per_100g: <from the source result>
       match_score: <initial estimate; select_best_candidate will rescore>
   - Call select_best_candidate(item=item, candidates=[...], min_score=0.5)
     separately for each source to get the single best candidate per source.
   - Skip a source if select_best_candidate returns None.

5. EVIDENCE + DELTAS
   - For each selected candidate, call compute_per_nutrient_delta(
       reported=item.nutrition_per_100g,
       observed=candidate.nutrition_per_100g
     ).
   - Call verdict_from_deltas(deltas=..., match_tolerance_pct=5.0,
       minor_tolerance_pct=15.0, absolute_floor_g=0.5).

6. ATWATER CHECK
   - Call check_atwater_consistency(
       nutrition=item.nutrition_per_100g,
       tolerance_pct=15.0
     ).

7. OVERALL VERDICT + CONFIDENCE
   - Item verdict = worst-case across all source bundles.
   - Confidence:
       HIGH  — ≥2 sources, item verdict is MATCH or MINOR_DISCREPANCY
       MEDIUM — ≥2 sources with disagreement, or 1 source with MAJOR_DISCREPANCY
       LOW   — only 1 source with MATCH/MINOR, or no data from any source

8. PROPOSED CORRECTION
   - If item verdict is MAJOR_DISCREPANCY and a USDA Foundation candidate was found,
     set proposed_correction to that candidate's nutrition_per_100g.

9. SUMMARY
   - Call format_human_summary(
       item=item,
       verdict_bundle=<worst-case bundle>,
       evidence_count=<number of selected candidates>
     ).

10. RETURN a complete ItemVerification with ALL fields populated.
"""

# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

_IO_TOOLS = [
    search_usda,
    get_usda_food,
    lookup_off_by_barcode,
    search_off_by_name,
    search_tavily,
]

_COMPUTE_TOOLS = [
    select_best_candidate_tool,
    compute_per_nutrient_delta_tool,
    verdict_from_deltas_tool,
    check_atwater_consistency_tool,
    format_human_summary_tool,
]


def build_verifier_agent(settings: Settings) -> Agent[VerifierContext]:
    """Build the verifier Agent with all tools and output guardrails.

    The returned agent has ``output_type=ItemVerification`` and two output
    guardrails:

    - :func:`~.guardrails.atwater_output_guardrail.atwater_output_guardrail`:
      re-derives the Atwater consistency flag to catch hallucination.
    - :func:`~.guardrails.schema_output_guardrail.schema_output_guardrail`:
      enforces structural invariants (non-empty evidence, non-blank summary).

    Args:
        settings: Application settings providing the model pin.

    Returns:
        Configured ``Agent[VerifierContext]`` ready for use with
        ``Runner.run(..., context=VerifierContext(...))``.
    """
    return Agent(
        name="verifier",
        instructions=_VERIFIER_INSTRUCTIONS,
        tools=_IO_TOOLS + _COMPUTE_TOOLS,
        output_type=ItemVerification,
        output_guardrails=[atwater_output_guardrail, schema_output_guardrail],
        model=settings.OPENAI_MODEL,
    )
