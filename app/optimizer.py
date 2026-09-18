"""Linear-programming battery/grid optimizer (Problem Statement Sections 05.2, 09).

Formulates the 24-hour schedule as an LP and solves it with CBC (via PuLP) to find the
true minimum-cost grid usage subject to every GridWise energy rule and every applicable
operator directive. A tiny activity penalty on charge/discharge breaks ties so the
solver never reports simultaneous charge-and-discharge in the same hour when it isn't
needed to satisfy a constraint.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

import pulp

from .schemas import BatteryConfig, HourEntry, HourlyPlanEntry

_ACTIVITY_PENALTY = 1e-6
_ZERO_TOLERANCE = 1e-6


class OptimizationInfeasibleError(RuntimeError):
    """Raised when no feasible 24-hour schedule exists for the given inputs."""


def solve_schedule(
    hours: List[HourEntry],
    battery: BatteryConfig,
    effective_solar: Dict[int, float],
    min_reserve: Dict[int, float],
    no_charge_hours: Set[int],
    no_discharge_hours: Set[int],
    max_grid: Dict[int, float],
) -> Tuple[List[HourlyPlanEntry], float, float, float]:
    hour_by_index = {entry.hour: entry for entry in hours}
    all_hours = list(range(24))

    problem = pulp.LpProblem("gridwise_schedule", pulp.LpMinimize)

    grid = {
        h: pulp.LpVariable(f"grid_{h}", lowBound=0, upBound=max_grid.get(h))
        for h in all_hours
    }
    solar_used = {
        h: pulp.LpVariable(f"solar_used_{h}", lowBound=0, upBound=max(effective_solar[h], 0.0))
        for h in all_hours
    }
    charge = {
        h: pulp.LpVariable(
            f"charge_{h}",
            lowBound=0,
            upBound=0.0 if h in no_charge_hours else battery.max_charge_kwh_per_hour,
        )
        for h in all_hours
    }
    discharge = {
        h: pulp.LpVariable(
            f"discharge_{h}",
            lowBound=0,
            upBound=0.0 if h in no_discharge_hours else battery.max_discharge_kwh_per_hour,
        )
        for h in all_hours
    }
    energy_after = {
        h: pulp.LpVariable(
            f"energy_after_{h}",
            lowBound=min_reserve[h],
            upBound=battery.capacity_kwh,
        )
        for h in all_hours
    }

    for h in all_hours:
        demand = hour_by_index[h].demand_kwh
        problem += grid[h] + solar_used[h] + discharge[h] == demand + charge[h], f"balance_{h}"

        previous_energy = battery.initial_energy_kwh if h == 0 else energy_after[h - 1]
        problem += energy_after[h] == previous_energy + charge[h] - discharge[h], f"transition_{h}"

    problem += energy_after[23] == battery.initial_energy_kwh, "end_of_day_neutrality"

    grid_cost = pulp.lpSum(grid[h] * hour_by_index[h].tariff_bdt_per_kwh for h in all_hours)
    activity_penalty = _ACTIVITY_PENALTY * pulp.lpSum(charge[h] + discharge[h] for h in all_hours)
    problem += grid_cost + activity_penalty

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[status] != "Optimal":
        raise OptimizationInfeasibleError(
            f"Solver returned status '{pulp.LpStatus[status]}' for this scenario"
        )

    hourly_plan: List[HourlyPlanEntry] = []
    total_grid_kwh = 0.0
    total_cost_bdt = 0.0
    peak_grid_kwh = 0.0

    for h in all_hours:
        grid_value = _clean(grid[h].value())
        solar_value = _clean(solar_used[h].value())
        net_battery = _clean(charge[h].value()) - _clean(discharge[h].value())
        energy_value = _clean(energy_after[h].value())

        if abs(net_battery) < _ZERO_TOLERANCE:
            action = "idle"
            magnitude = 0.0
        elif net_battery > 0:
            action = "charge"
            magnitude = net_battery
        else:
            action = "discharge"
            magnitude = -net_battery

        hourly_plan.append(
            HourlyPlanEntry(
                hour=h,
                grid_kwh=round(grid_value, 6),
                solar_used_kwh=round(solar_value, 6),
                battery_action=action,
                battery_kwh=round(magnitude, 6),
                battery_energy_after_kwh=round(energy_value, 6),
            )
        )

        tariff = hour_by_index[h].tariff_bdt_per_kwh
        total_grid_kwh += grid_value
        total_cost_bdt += grid_value * tariff
        peak_grid_kwh = max(peak_grid_kwh, grid_value)

    return (
        hourly_plan,
        round(total_grid_kwh, 6),
        round(total_cost_bdt, 6),
        round(peak_grid_kwh, 6),
    )


def _clean(value: float) -> float:
    if value is None:
        return 0.0
    if abs(value) < _ZERO_TOLERANCE:
        return 0.0
    return value
