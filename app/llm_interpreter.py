"""LLM-backed operator-note interpretation (Problem Statement Sections 02, 04, 08).

This is the ONLY place a language model is called. Its output is untrusted structured
data -- app.guardrails.validate_and_normalize() must always run on whatever this module
returns before it reaches the optimizer.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .schemas import DIRECTIVE_TYPES, BatteryConfig

# Loads a local .env file if present (no-op in production/Docker where real env vars
# are injected directly). Must run before the module-level os.environ.get() calls below.
load_dotenv()

logger = logging.getLogger("gridwise.llm")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LLM_CALL_TIMEOUT_MS = int(os.environ.get("LLM_CALL_TIMEOUT_MS", "10000"))

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


_RESPONSE_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "directives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "note_index": {"type": "integer"},
                    "applies": {"type": "boolean"},
                    "directive_type": {"type": "string", "enum": list(DIRECTIVE_TYPES)},
                    "hours": {"type": "array", "items": {"type": "integer"}},
                    "factor": {"type": "number"},
                    "minimum_energy_kwh": {"type": "number"},
                    "max_grid_kwh": {"type": "number"},
                    "explanation": {"type": "string"},
                },
                "required": ["note_index", "applies", "directive_type", "explanation"],
            },
        }
    },
    "required": ["directives"],
}

_SYSTEM_INSTRUCTION = """You are the operator-note interpreter for the GridWise campus \
energy scheduling system. You convert short natural-language operator notes into strict, \
machine-checkable directives for a downstream energy optimizer. You never perform the \
optimization yourself, and you never invent demand, solar, tariff, or battery values.

Supported directive types (use exactly one per note, or "no_op" if the note does not \
affect today's 24-hour energy schedule):

- solar_reduction: usable solar is reduced during specific hours.
  fields: hours (list of int), factor (fraction of solar that REMAINS usable, 0..1).
  Example: "drop to 20%" or "an 80% reduction" both mean factor = 0.2.
- minimum_battery_reserve: battery energy must stay at or above a level during specific hours.
  fields: hours (list of int), minimum_energy_kwh (absolute kWh; convert percentages of
  capacity into kWh using the battery capacity given in the prompt).
- no_charge_window: battery charging is unavailable during specific hours.
  fields: hours (list of int).
- no_discharge_window: battery discharging is unavailable during specific hours.
  fields: hours (list of int).
- max_grid_window: grid import may not exceed a stated kWh amount during specific hours.
  fields: hours (list of int), max_grid_kwh (number).
- no_op: the note is a distractor / does not affect the energy schedule. No fields needed.

Hour convention: hours are whole-hour integers 0-23. A window is start-inclusive and
end-exclusive: "1 PM to 3 PM" or "13:00 to 15:00" means hours [13, 14], NOT [13, 14, 15].
"1 AM" = hour 1, "midnight"/"12 AM" = hour 0, "noon"/"12 PM" = hour 12.

Rules:
- Produce exactly one directive object per operator note, in the same order as given.
- Only use the six directive types listed above. Never invent a new type.
- For no_op: applies = false.
- For every other directive type: applies = true, and you must fill in the fields that
  type requires (hours, plus the one numeric field that type needs).
- Notes may paraphrase the same underlying rule in different words, percentages, or
  equivalent numeric descriptions -- resolve them to the same directive semantics.
- Do not change or invent demand, solar, tariff, or battery parameters yourself. Only
  emit the directive fields defined above; the optimizer applies them deterministically.
- Respond ONLY with JSON matching the required schema. No prose, no markdown fences.
"""

_PROMPT_TEMPLATE = """Battery capacity for this scenario: {battery_capacity_kwh} kWh.
(Use this only to convert relative/percentage reserve language into absolute kWh for
minimum_battery_reserve directives.)

There are {num_notes} operator note(s), indexed from 0. Interpret each one:

{notes_block}

Return the JSON object with a "directives" array containing exactly {num_notes} entries,
one per note_index, in ascending order.
"""


def interpret_operator_notes(
    operator_notes: List[str], battery: BatteryConfig
) -> List[Dict[str, Any]]:
    """Calls the LLM to interpret operator notes.

    Returns a list of raw, UNTRUSTED dicts -- callers must run them through
    app.guardrails.validate_and_normalize() before use. Raises on transport/provider
    failure so the caller can decide whether to retry or fail safely.
    """
    client = _get_client()

    notes_block = "\n".join(f'{idx}: "{note}"' for idx, note in enumerate(operator_notes))
    user_prompt = _PROMPT_TEMPLATE.format(
        battery_capacity_kwh=battery.capacity_kwh,
        num_notes=len(operator_notes),
        notes_block=notes_block,
    )

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=_RESPONSE_JSON_SCHEMA,
            temperature=0,
            # The SDK defaults to up to 5 internal retries with backoff on failure,
            # which under a provider outage can silently blow past our own request
            # timeout budget. Bound each HTTP attempt explicitly and let our own
            # single retry in main._interpret_with_retry own the retry decision.
            http_options=types.HttpOptions(
                timeout=LLM_CALL_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        ),
    )

    raw_text = response.text
    try:
        parsed = json.loads(raw_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.warning("LLM returned non-JSON output; treating as empty directive list")
        return []

    directives = parsed.get("directives") if isinstance(parsed, dict) else None
    if not isinstance(directives, list):
        logger.warning("LLM JSON output missing a 'directives' array: %r", parsed)
        return []
    return directives
