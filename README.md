# snaq-verify

Deterministic, agent-orchestrated nutrition verification system. Take-home for SNAQ.

> Full README written by the orchestrator at the end of Phase 3. Until then, see
> `docs/` and the plan file.

## Quickstart (single command)

```bash
cp .env.example .env   # paste USDA_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY
docker compose up      # builds image, runs verify + eval, writes output/*.json
```

## Local development

```bash
make setup
make all
```
