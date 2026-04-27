"""Top-level verification report."""

from datetime import datetime

from pydantic import BaseModel, Field

from snaq_verify.domain.models.item_verification import ItemVerification


class RunMetadata(BaseModel):
    """Provenance for a single run.

    `timestamp` is the only field that varies between runs on the same input
    when the cache is warm — used by the determinism check.
    """

    timestamp: datetime
    input_count: int
    flag_count: int  # items with verdict in {minor_discrepancy, major_discrepancy}
    model: str  # OPENAI_MODEL pin used by the verifier and judge
    snaq_verify_version: str = "0.1.0"


class VerificationReport(BaseModel):
    """The deliverable written to `verification_report.json`."""

    metadata: RunMetadata
    items: list[ItemVerification] = Field(default_factory=list)
