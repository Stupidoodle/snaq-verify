"""LoadInputStep — reads food_items.json and populates state.items."""

import json
from pathlib import Path

from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.pipeline_state import PipelineState
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.pipeline_step_port import PipelineStep


class LoadInputStep(PipelineStep):
    """Read `state.input_path` as a JSON array of FoodItem objects.

    Validates each item against the `FoodItem` Pydantic schema.  Raises
    `ValueError` if `state.input_path` is None, `FileNotFoundError` if the
    file does not exist, and `ValueError` for malformed JSON or schema
    violations.
    """

    def __init__(self, logger: LoggerPort) -> None:
        """Construct the step.

        Args:
            logger: Structured logger for step events.
        """
        self._logger = logger

    @property
    def name(self) -> str:
        """Step identifier."""
        return "load_input"

    async def run(self, state: PipelineState) -> PipelineState:
        """Read the input file and populate `state.items`.

        Args:
            state: Pipeline state; `input_path` must be set.

        Returns:
            Updated state with `items` populated.

        Raises:
            ValueError: If `input_path` is None or the JSON is invalid.
            FileNotFoundError: If the input file does not exist.
        """
        if state.input_path is None:
            raise ValueError("LoadInputStep requires state.input_path to be set")

        path: Path = state.input_path
        self._logger.info("load_input.reading", path=str(path))

        try:
            raw = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._logger.error("load_input.file_not_found", path=str(path))
            raise

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in {path}, got {type(data).__name__}"
            )

        items = [FoodItem.model_validate(entry) for entry in data]
        state.items = items
        self._logger.info("load_input.done", count=len(items), path=str(path))
        return state
