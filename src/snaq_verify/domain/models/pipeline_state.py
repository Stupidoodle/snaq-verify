"""Pipeline state — mutable bag passed step-to-step."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from snaq_verify.domain.models.eval_models import EvalReport, GroundTruthEntry
from snaq_verify.domain.models.food_item import FoodItem
from snaq_verify.domain.models.item_verification import ItemVerification
from snaq_verify.domain.models.verification_report import VerificationReport


class PipelineState(BaseModel):
    """The single object handed step-to-step in the runner.

    Each step reads the slices it needs and writes its output. Adding a new
    step = adding a slice + registering the step in bootstrap.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # input
    input_path: Path | None = None
    output_path: Path | None = None
    eval_output_path: Path | None = None
    ground_truth_path: Path | None = None

    # accumulated artifacts
    items: list[FoodItem] = Field(default_factory=list)
    verifications: list[ItemVerification] = Field(default_factory=list)
    report: VerificationReport | None = None

    # eval-only artifacts
    ground_truth: list[GroundTruthEntry] = Field(default_factory=list)
    eval_report: EvalReport | None = None
