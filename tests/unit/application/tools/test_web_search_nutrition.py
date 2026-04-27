"""Unit tests for the web_search_nutrition function tool."""

import pytest

from snaq_verify.domain.models.source_lookup import WebSnippet
from tests.fakes.fake_tavily_client import FakeTavilyClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CHICKEN_SNIPPET = WebSnippet(
    url="https://nutritiondata.self.com/chicken",
    title="Chicken Breast Nutrition",
    content="Chicken breast: 23g protein, 1g fat per 100g.",
    score=0.95,
)

_SALMON_SNIPPET = WebSnippet(
    url="https://nutritiondata.self.com/salmon",
    title="Salmon Nutrition",
    content="Atlantic salmon: 20g protein, 13g fat per 100g.",
    score=0.88,
)


# ---------------------------------------------------------------------------
# Helper: configure the tool with a fake client before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def configure_tool(request: pytest.FixtureRequest) -> None:  # type: ignore[return]
    """Inject a FakeTavilyClient into the tool before each test."""
    import snaq_verify.application.tools.web_search_nutrition as mod

    fake = getattr(request, "param", None) or FakeTavilyClient(
        responses={
            "chicken breast nutrition per 100g": [_CHICKEN_SNIPPET],
            "atlantic salmon nutrition per 100g": [_SALMON_SNIPPET],
        }
    )
    mod.configure_web_search(fake)
    yield
    # Reset after test
    mod.configure_web_search(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_snippets_for_known_query() -> None:
    """Tool returns snippets from the underlying client."""
    from snaq_verify.application.tools.web_search_nutrition import web_search_nutrition

    results = await web_search_nutrition("chicken breast nutrition per 100g")
    assert len(results) == 1
    assert results[0].url == _CHICKEN_SNIPPET.url
    assert results[0].title == _CHICKEN_SNIPPET.title


@pytest.mark.asyncio
async def test_returns_empty_list_for_unknown_query() -> None:
    """Tool returns an empty list when the client has no results."""
    from snaq_verify.application.tools.web_search_nutrition import web_search_nutrition

    results = await web_search_nutrition("xyzzy foobarbaz")
    assert results == []


@pytest.mark.asyncio
async def test_custom_max_results_forwarded() -> None:
    """max_results is passed through to the underlying client."""
    import snaq_verify.application.tools.web_search_nutrition as mod

    many_snippets = [_CHICKEN_SNIPPET, _SALMON_SNIPPET]
    fake = FakeTavilyClient(responses={"nutrition query": many_snippets})
    mod.configure_web_search(fake)

    from snaq_verify.application.tools.web_search_nutrition import web_search_nutrition

    results = await web_search_nutrition("nutrition query", max_results=1)
    # FakeTavilyClient truncates to max_results
    assert len(results) == 1


@pytest.mark.asyncio
async def test_default_max_results_is_five() -> None:
    """Default max_results value is 5."""
    import snaq_verify.application.tools.web_search_nutrition as mod

    fake = FakeTavilyClient(responses={"query": [_CHICKEN_SNIPPET]})
    mod.configure_web_search(fake)

    from snaq_verify.application.tools.web_search_nutrition import web_search_nutrition

    await web_search_nutrition("query")
    assert fake.calls[-1] == ("query", 5)


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_raises_when_not_configured() -> None:
    """Tool raises RuntimeError when no client has been configured."""
    import snaq_verify.application.tools.web_search_nutrition as mod

    mod.configure_web_search(None)  # type: ignore[arg-type]

    from snaq_verify.application.tools.web_search_nutrition import web_search_nutrition

    with pytest.raises(RuntimeError, match="not configured"):
        await web_search_nutrition("any query")


@pytest.mark.asyncio
async def test_propagates_client_errors() -> None:
    """Errors from the underlying client propagate unchanged."""
    import snaq_verify.application.tools.web_search_nutrition as mod

    fake = FakeTavilyClient(raise_on="bad query")
    mod.configure_web_search(fake)

    from snaq_verify.application.tools.web_search_nutrition import web_search_nutrition

    with pytest.raises(RuntimeError, match="simulated error"):
        await web_search_nutrition("bad query")
