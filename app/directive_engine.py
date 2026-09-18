"""Applies validated directives to the optimization inputs (Problem Statement Section 05.3).

This module performs the deterministic "Guardrail Validator -> Math Optimizer" step of
the end-to-end flow in Section 03: it never talks to the LLM and never invents a rule,
it only mechanically applies already-validated DirectiveInterpretation objects.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .schemas import BatteryConfig, DirectiveInterpretation, HourEntry


def apply_directives(
    hours: List[HourEntry],
    battery: BatteryConfig,
    directives: List[DirectiveInterpretation],
) -> Tuple[Dict[int, float], Dict[int, float], Set[int], Set[int], Dict[int, float]]:
    """Returns (effective_solar, min_reserve, no_charge_hours, no_discharge_hours, max_grid)."""

    base_solar = {entry.hour: entry.solar_kwh for entry in hours}
    solar_factor = {h: 1.0 for h in range(24)}
    min_reserve = {h: battery.minimum_energy_kwh for h in range(24)}
    no_charge_hours: Set[int] = set()
    no_discharge_hours: Set[int] = set()
    max_grid: Dict[int, float] = {}

    for directive in directives:
        if not directive.applies or directive.directive_type == "no_op":
            continue

        adjustment = directive.structured_adjustment or {}
        affected_hours = adjustment.get("hours", [])

        if directive.directive_type == "solar_reduction":
            factor = float(adjustment["factor"])
            for h in affected_hours:
                solar_factor[h] *= factor

        elif directive.directive_type == "minimum_battery_reserve":
            reserve = float(adjustment["minimum_energy_kwh"])
            for h in affected_hours:
                min_reserve[h] = max(min_reserve[h], reserve)

        elif directive.directive_type == "no_charge_window":
            no_charge_hours.update(affected_hours)

        elif directive.directive_type == "no_discharge_window":
            no_discharge_hours.update(affected_hours)

        elif directive.directive_type == "max_grid_window":
            cap = float(adjustment["max_grid_kwh"])
            for h in affected_hours:
                max_grid[h] = min(max_grid.get(h, cap), cap)

    effective_solar = {h: base_solar[h] * solar_factor[h] for h in range(24)}
    return effective_solar, min_reserve, no_charge_hours, no_discharge_hours, max_grid
