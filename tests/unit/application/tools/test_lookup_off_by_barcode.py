"""Unit tests for the lookup_off_by_barcode tool factory."""

import pytest
from agents import FunctionTool

from snaq_verify.application.tools.lookup_off_by_barcode import make_lookup_off_by_barcode
from snaq_verify.domain.models.source_lookup import OFFProduct
from tests.fakes.fake_open_food_facts_client import FakeOpenFoodFactsClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BARCODE = "3017620422003"
_FAGE_BARCODE = "5200435000027"

_NUTELLA = OFFProduct(
    code=_BARCODE,
    product_name="Nutella",
    brands="Ferrero",
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_returns_product_when_found() -> None:
    """Raw function returns the OFFProduct when the barcode exists in OFF."""
    client = FakeOpenFoodFactsClient(barcode_map={_BARCODE: _NUTELLA})
    fn, _ = make_lookup_off_by_barcode(client)

    result = await fn(_BARCODE)

    assert result is not None
    assert isinstance(result, OFFProduct)
    assert result.code == _BARCODE
    assert result.product_name == "Nutella"


@pytest.mark.asyncio
async def test_returns_none_for_unknown_barcode() -> None:
    """Fage barcode (not in OFF) → raw function returns None, not an exception."""
    # barcode_map is empty: any barcode → None
    client = FakeOpenFoodFactsClient()
    fn, _ = make_lookup_off_by_barcode(client)

    result = await fn(_FAGE_BARCODE)

    assert result is None


@pytest.mark.asyncio
async def test_returns_none_when_explicitly_mapped_to_none() -> None:
    """Barcode explicitly mapped to None → returns None."""
    client = FakeOpenFoodFactsClient(barcode_map={_FAGE_BARCODE: None})
    fn, _ = make_lookup_off_by_barcode(client)

    result = await fn(_FAGE_BARCODE)

    assert result is None


@pytest.mark.asyncio
async def test_tracks_barcode_call() -> None:
    """The fake records the barcode that was looked up."""
    client = FakeOpenFoodFactsClient(barcode_map={_BARCODE: _NUTELLA})
    fn, _ = make_lookup_off_by_barcode(client)

    await fn(_BARCODE)

    assert client.barcode_calls == [_BARCODE]


@pytest.mark.asyncio
async def test_propagates_client_error() -> None:
    """Network-level errors from the client propagate out of the raw function."""
    client = FakeOpenFoodFactsClient(raise_on_barcode=_BARCODE)
    fn, _ = make_lookup_off_by_barcode(client)

    with pytest.raises(RuntimeError):
        await fn(_BARCODE)


def test_tool_export_is_function_tool() -> None:
    """Factory exports a FunctionTool suitable for Agent(tools=[...])."""
    client = FakeOpenFoodFactsClient()
    _, tool = make_lookup_off_by_barcode(client)

    assert isinstance(tool, FunctionTool)
    assert tool.name == "lookup_off_by_barcode"
