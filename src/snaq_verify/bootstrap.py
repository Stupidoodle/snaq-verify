"""Composition root — wires every concrete adapter behind its port.

This is the *only* place real adapters are instantiated. Tests build
containers from the fake adapters in `tests/fakes/`; runtime code in
`cli/main.py` calls `Bootstrap.build()` to obtain a wired `Container`.

Lazy imports inside `build()` mean this module is importable before
Phase 2 adapters land — only calling `build()` requires every adapter
file to exist. Teammates implementing adapters do not need to modify
this file beyond toggling on their own adapter line.
"""

from dataclasses import dataclass

from snaq_verify.application.pipeline.runner import PipelineRunner
from snaq_verify.core.config import Settings, get_settings
from snaq_verify.domain.ports.cache_port import CachePort
from snaq_verify.domain.ports.judge_agent_port import JudgeAgentPort
from snaq_verify.domain.ports.logger_port import LoggerPort
from snaq_verify.domain.ports.open_food_facts_client_port import (
    OpenFoodFactsClientPort,
)
from snaq_verify.domain.ports.tavily_client_port import TavilyClientPort
from snaq_verify.domain.ports.usda_client_port import USDAClientPort
from snaq_verify.domain.ports.verifier_agent_port import VerifierAgentPort


@dataclass
class Container:
    """Holds every wired dependency for the application."""

    settings: Settings
    logger: LoggerPort
    cache: CachePort
    usda: USDAClientPort
    off: OpenFoodFactsClientPort
    tavily: TavilyClientPort
    verifier_agent: VerifierAgentPort
    judge_agent: JudgeAgentPort
    runner: PipelineRunner


class Bootstrap:
    """Build the wired `Container`."""

    @staticmethod
    def build(settings: Settings | None = None) -> Container:
        """Build a fully wired `Container` from real adapters.

        Args:
            settings: Optional override; defaults to `get_settings()`.

        Returns:
            A `Container` ready for use by the CLI.
        """
        settings = settings or get_settings()

        # Lazy adapter imports — keeps this module importable while Phase 2
        # adapters are still in flight. Each teammate fills in their import
        # + constructor call in their own commit.
        from snaq_verify.infrastructure.observability.structlog_logger import (
            StructlogLogger,
        )

        logger: LoggerPort = StructlogLogger(level=settings.LOG_LEVEL)

        from snaq_verify.infrastructure.cache.file_cache import FileCache

        cache: CachePort = FileCache(
            cache_dir=settings.CACHE_DIR,
            ttl_seconds=settings.CACHE_TTL_DAYS * 86400,
            logger=logger,
        )

        from snaq_verify.infrastructure.sources.usda_client import USDAClient

        usda: USDAClientPort = USDAClient(
            settings=settings, logger=logger, cache=cache,
        )

        from snaq_verify.infrastructure.sources.open_food_facts_client import (
            OpenFoodFactsClient,
        )

        off: OpenFoodFactsClientPort = OpenFoodFactsClient(
            settings=settings, logger=logger, cache=cache,
        )

        from snaq_verify.infrastructure.sources.tavily_client import TavilyClient

        tavily: TavilyClientPort = TavilyClient(
            settings=settings, logger=logger, cache=cache,
        )

        from snaq_verify.infrastructure.agents.verifier_agent_adapter import (
            VerifierAgentAdapter,
        )

        verifier_agent: VerifierAgentPort = VerifierAgentAdapter(
            settings=settings,
            logger=logger,
            usda=usda,
            off=off,
            tavily=tavily,
        )

        from snaq_verify.infrastructure.agents.judge_agent_adapter import (
            JudgeAgentAdapter,
        )

        judge_agent: JudgeAgentPort = JudgeAgentAdapter(
            settings=settings, logger=logger,
        )

        runner = PipelineRunner(logger=logger)

        return Container(
            settings=settings,
            logger=logger,
            cache=cache,
            usda=usda,
            off=off,
            tavily=tavily,
            verifier_agent=verifier_agent,
            judge_agent=judge_agent,
            runner=runner,
        )
