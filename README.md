# GridWise LLM-Assisted Energy Optimization Service

BUP CSE Fest 2026 Hackathon — Online Preliminary submission for the "Smart Campus Energy
Optimization Challenge / LLM-Assisted Operator Directive Interpretation" problem.

This service exposes `GET /health` and `POST /optimize-energy`. It interprets 1-3
natural-language operator notes with a language model, deterministically guardrails
that interpretation, applies it to a linear-programming optimizer, and returns a valid,
cost-minimizing 24-hour battery/grid schedule. See the official Problem Statement and
Participant Guide (in the parent folder) for the full contract this implements.

## Architecture

```
Energy Data + Operator Notes
        |
        v
  LLM Interpreter        app/llm_interpreter.py   (Google Gemini; the ONLY LLM call)
        |  (untrusted structured JSON)
        v
  Guardrail Validator     app/guardrails.py        (deterministic; safe no_op fallback)
        |  (trusted DirectiveInterpretation list)
        v
  Directive Engine        app/directive_engine.py  (turns directives into optimizer inputs)
        |
        v
  Math Optimizer          app/optimizer.py         (PuLP + CBC linear program)
        |  (hourly_plan)
        v
  API Response            app/main.py              (FastAPI; schema-exact JSON)
```

- **LLM role**: `app/llm_interpreter.py` sends the operator notes plus the directive
  taxonomy (from Problem Statement Section 04) to Gemini with a strict JSON response
  schema, and asks it to classify each note as one of the six supported directive types
  or `no_op`. The LLM's output is never trusted directly — it is the ONLY input to the
  next stage, and that stage is deterministic.
- **Guardrails**: `app/guardrails.py` validates every field the LLM returned (allowed
  directive type, hour range/uniqueness/order, `factor` in `[0,1]`, reserve/cap values
  finite and non-negative, `applies` semantics). Anything invalid, missing, or duplicated
  is safely downgraded to `no_op` for that note instead of crashing or inventing a rule.
- **Optimizer**: `app/optimizer.py` builds one LP per request (variables: grid import,
  solar used, battery charge, battery discharge, battery energy-after, per hour) and
  solves it with the CBC solver bundled in PuLP to minimize
  `sum(grid_kwh[h] * tariff_bdt_per_kwh[h])` subject to every GridWise energy rule
  (Section 09) and every applied directive (Section 05.3), including end-of-day battery
  neutrality.

## Requirements

- Python 3.11+
- A Google Gemini API key (https://ai.google.dev/) with access to a Gemini model that
  supports JSON response schemas (default: `gemini-flash-lite-latest`).

## Environment variables

| Variable | Required | Default | Meaning |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | API key for Google Gemini. Used only for operator-note interpretation. |
| `GEMINI_MODEL` | No | `gemini-flash-lite-latest` | Gemini model id. |
| `PORT` | No | `8080` | Port the service listens on. |
| `LOG_LEVEL` | No | `INFO` | Python logging level. |
| `REQUEST_TIMEOUT_SECONDS` | No | `25` | Internal hard budget per `/optimize-energy` request before returning a controlled 500 (kept under the judge's 30s timeout). |

Never commit a real `.env` file. `.env.example` documents the variable names only.

## Local quickstart (clean environment)

```bash
git clone <this-repo-url>
cd gridwise-service

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY=<your key>

uvicorn app.main:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
# {"status":"ok"}
```

Sample request:

```bash
curl -X POST http://localhost:8080/optimize-energy \
  -H "Content-Type: application/json" \
  -d @../BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json
```

(The public sample file wraps each case as `{"input": {...}, "expected_output": {...}}`;
use the harness below to drive every case's `input` object automatically.)

## Running the public sample cases

With the service running locally on port 8080:

```bash
pip install -r requirements-dev.txt
python tests/run_public_samples.py --base-url http://localhost:8080
```

This posts every case from `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json` to
`/optimize-energy`, checks the response schema, compares the returned
`directive_interpretation` against the case's reference interpretation, and
independently replays `hourly_plan` against energy balance, effective solar, battery
bounds/rate limits, directive constraints, and end-of-day neutrality — the same class of
checks the hidden judge performs. It prints a PASS/FAIL summary per case.

## Docker fallback

```bash
docker build -t gridwise-service .
docker run --rm -p 8080:8080 \
  -e GEMINI_API_KEY=<your key> \
  -e GEMINI_MODEL=gemini-flash-lite-latest \
  gridwise-service

curl http://localhost:8080/health
```

The image binds to `0.0.0.0`, exposes the port from `$PORT` (default `8080`), and
contains no baked-in secrets — the API key is supplied at `docker run` time.

## Dependencies

- [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) — HTTP API.
- [Pydantic v2](https://docs.pydantic.dev/) — request/response schema validation.
- [google-genai](https://pypi.org/project/google-genai/) — Google Gemini SDK, used only
  in `app/llm_interpreter.py` for operator-note interpretation.
- [PuLP](https://github.com/coin-or/pulp) (with the bundled CBC solver) — the 24-hour
  linear-programming schedule optimizer.

## Known limitations

- The optimizer treats simultaneous charge-and-discharge in the same hour as strictly
  non-beneficial and applies a negligible tie-breaking penalty to avoid reporting it;
  this does not change the optimal cost.
- If the Gemini API is unavailable after one retry, the service does not fail the
  request — it safely falls back to `no_op` for every note (Problem Statement Section
  08, "SAFE FAILURE") rather than crashing or guessing a directive. This trades
  interpretation credit for availability; a working `GEMINI_API_KEY` is required to score
  on LLM Directive Interpretation.
- `tests/run_public_samples.py` is a local development aid, not the hidden judge
  harness; public cases are not the hidden scoring set.

## Secret handling

No API keys, tokens, or `.env` files are committed to this repository (see `.gitignore`).
The Docker image takes `GEMINI_API_KEY` as a runtime environment variable only. Error
responses never include raw exception messages or stack traces.
