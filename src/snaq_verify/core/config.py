"""Application settings loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    Loaded from environment variables (preferred) with fallback to a `.env`
    file. All field names are case-sensitive and uppercase, matching env vars
    1:1 to keep the contract obvious.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- API credentials ----------------------------------------------------
    USDA_API_KEY: str
    OPENAI_API_KEY: str
    TAVILY_API_KEY: str

    # --- LLM model pin ------------------------------------------------------
    # Pin a single model so reruns produce stable structured outputs.
    OPENAI_MODEL: str = "gpt-5.4-mini"

    # --- HTTP client config -------------------------------------------------
    HTTP_TIMEOUT_SECONDS: float = 15.0

    # --- USDA FoodData Central ---------------------------------------------
    USDA_BASE_URL: str = "https://api.nal.usda.gov/fdc/v1"

    # --- Open Food Facts ---------------------------------------------------
    # User-Agent header is mandatory per OFF policy. Format: AppName/Version (email).
    OFF_USER_AGENT: str = "snaq-verify/0.1 (bryan.tran053@gmail.com)"
    OFF_BASE_URL: str = "https://world.openfoodfacts.org"

    # --- Cache --------------------------------------------------------------
    CACHE_DIR: Path = Path(".cache")
    CACHE_TTL_DAYS: int = 30

    # --- Verdict thresholds (per-nutrient) ---------------------------------
    # |relative delta| <= MATCH_TOLERANCE_PCT ............ match
    # MATCH_TOLERANCE_PCT < |..| <= MINOR_TOLERANCE_PCT .. minor_discrepancy
    # |..| > MINOR_TOLERANCE_PCT ......................... major_discrepancy
    MATCH_TOLERANCE_PCT: float = 5.0
    MINOR_TOLERANCE_PCT: float = 15.0

    # Atwater consistency: |kcal_reported - (4P + 4C + 9F)| / kcal_reported.
    ATWATER_TOLERANCE_PCT: float = 15.0

    # Below this absolute value (grams) we suppress relative-delta verdicts to
    # avoid false majors on near-zero nutrients (e.g., 0.0 vs 0.3 g fiber).
    ABSOLUTE_FLOOR_G: float = 0.5

    # Candidate match score below which we treat a source lookup as a miss.
    MIN_CANDIDATE_SCORE: float = 0.5

    # --- Logging ------------------------------------------------------------
    LOG_LEVEL: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Uses lru_cache so repeated calls in the same process share the same
    instance and `.env` parsing happens exactly once.
    """
    return Settings()  # type: ignore[call-arg]
