"""Deterministic validation of untrusted LLM output (Problem Statement Section 08).

The LLM's raw output is treated as untrusted structured data. Every entry is checked
against the exact rules in Section 04/05/08 before it is ever handed to the optimizer.
Anything that fails validation is safely downgraded to a no_op for that note instead of
crashing the service or inventing a new directive (Section 08, "SAFE FAILURE").
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .schemas import BatteryConfig, DirectiveInterpretation, DIRECTIVE_TYPES

logger = logging.getLogger("gridwise.guardrails")

_REAL_DIRECTIVE_TYPES = {t for t in DIRECTIVE_TYPES if t != "no_op"}


def _fallback_no_op(note_index: int, reason: str) -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=note_index,
        applies=False,
        directive_type="no_op",
        structured_adjustment=None,
        explanation=f"Safe fallback to no_op: {reason}",
    )


def _normalize_hours(raw_hours: Any) -> Optional[List[int]]:
    if not isinstance(raw_hours, list) or len(raw_hours) == 0:
        return None
    try:
        ints = [int(h) for h in raw_hours]
    except (TypeError, ValueError):
        return None
    if any(h < 0 or h > 23 for h in ints):
        return None
    if len(set(ints)) != len(ints):
        return None
    return sorted(ints)


def _normalize_number(raw_value: Any) -> Optional[float]:
    if isinstance(raw_value, bool):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float("inf"), float("-inf")):  # NaN / inf guard
        return None
    return value


def _build_structured_adjustment(
    directive_type: str, entry: Dict[str, Any], battery: BatteryConfig
) -> Optional[Dict[str, Any]]:
    hours = _normalize_hours(entry.get("hours"))
    if hours is None:
        return None

    if directive_type == "solar_reduction":
        factor = _normalize_number(entry.get("factor"))
        if factor is None or not (0.0 <= factor <= 1.0):
            return None
        return {"hours": hours, "factor": factor}

    if directive_type == "minimum_battery_reserve":
        reserve = _normalize_number(entry.get("minimum_energy_kwh"))
        if reserve is None or reserve < 0 or reserve > battery.capacity_kwh:
            return None
        return {"hours": hours, "minimum_energy_kwh": reserve}

    if directive_type in ("no_charge_window", "no_discharge_window"):
        return {"hours": hours}

    if directive_type == "max_grid_window":
        cap = _normalize_number(entry.get("max_grid_kwh"))
        if cap is None or cap < 0:
            return None
        return {"hours": hours, "max_grid_kwh": cap}

    return None


def _validate_entry(
    entry: Dict[str, Any], battery: BatteryConfig
) -> Optional[DirectiveInterpretation]:
    if not isinstance(entry, dict):
        return None

    try:
        note_index = int(entry.get("note_index"))
    except (TypeError, ValueError):
        return None

    directive_type = entry.get("directive_type")
    if directive_type not in DIRECTIVE_TYPES:
        return None

    explanation = entry.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        explanation = "No explanation provided by the model."
    explanation = explanation.strip()[:500]

    if directive_type == "no_op":
        return DirectiveInterpretation(
            note_index=note_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation=explanation,
        )

    # Any non-no_op directive must be reported as applies = True (Section 05/08).
    # If the model reported applies = False for a real directive type, its output is
    # self-contradictory and untrustworthy -> treat the whole entry as invalid.
    if entry.get("applies") is not True:
        return None

    structured_adjustment = _build_structured_adjustment(directive_type, entry, battery)
    if structured_adjustment is None:
        return None

    return DirectiveInterpretation(
        note_index=note_index,
        applies=True,
        directive_type=directive_type,
        structured_adjustment=structured_adjustment,
        explanation=explanation,
    )


def validate_and_normalize(
    raw_entries: List[Dict[str, Any]], num_notes: int, battery: BatteryConfig
) -> List[DirectiveInterpretation]:
    """Validate raw LLM output and guarantee exactly one entry per note_index 0..N-1.

    Missing, duplicate, malformed, or out-of-range entries are safely replaced with
    a no_op fallback for that note rather than propagating bad data to the optimizer.
    """
    claimed: Dict[int, DirectiveInterpretation] = {}

    if isinstance(raw_entries, list):
        for raw_entry in raw_entries:
            try:
                validated = _validate_entry(raw_entry, battery)
            except Exception:
                logger.exception("Unexpected error validating LLM directive entry: %r", raw_entry)
                validated = None

            if validated is None:
                continue
            if validated.note_index < 0 or validated.note_index >= num_notes:
                continue
            if validated.note_index in claimed:
                # Duplicate mapping for the same note: first valid entry wins.
                continue
            claimed[validated.note_index] = validated
    else:
        logger.warning("LLM output was not a list of directive entries: %r", raw_entries)

    result: List[DirectiveInterpretation] = []
    for note_index in range(num_notes):
        if note_index in claimed:
            result.append(claimed[note_index])
        else:
            result.append(
                _fallback_no_op(note_index, "missing or invalid model output for this note")
            )
    return result
