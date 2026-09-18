#!/usr/bin/env python3
"""Local validation harness for the GridWise service against the organizer's public
sample cases (BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json).

Usage:
    python tests/run_public_samples.py [--base-url http://localhost:8080] [--samples PATH]

For each public case this script:
  1. POSTs case.input to <base_url>/optimize-energy.
  2. Checks the response against the required schema shape.
  3. Compares directive_interpretation (applies / directive_type / hours / key numeric
     field) against the case's reference interpretation.
  4. Independently replays hourly_plan using the SERVICE'S OWN reported directives to
     verify energy balance, effective solar, battery bounds/rate limits, directive
     constraints, and end-of-day neutrality -- the same checks the hidden judge runs.
  5. Prints a pass/fail summary and the recalculated cost vs. the reference cost.

This is a convenience script for local development, not the hidden judge harness.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests

TOLERANCE = 0.01


def load_cases(samples_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(samples_path.read_text(encoding="utf-8"))
    return data["cases"]


def close(a: float, b: float, tol: float = TOLERANCE) -> bool:
    return abs(a - b) <= tol


def check_interpretation(case_id: str, got: List[dict], expected: List[dict]) -> List[str]:
    problems = []
    got_by_index = {entry["note_index"]: entry for entry in got}

    expected_indices = sorted(e["note_index"] for e in expected)
    if expected_indices != list(range(len(expected))):
        problems.append(f"[{case_id}] reference case itself has irregular note_index set (ignored)")

    if sorted(got_by_index.keys()) != list(range(len(expected))):
        problems.append(
            f"[{case_id}] directive_interpretation must cover note_index 0..{len(expected) - 1} exactly once; got {sorted(got_by_index.keys())}"
        )
        return problems

    for exp in expected:
        idx = exp["note_index"]
        act = got_by_index[idx]

        if act["applies"] != exp["applies"]:
            problems.append(f"[{case_id}] note {idx}: applies mismatch (got {act['applies']}, expected {exp['applies']})")
            continue
        if act["directive_type"] != exp["directive_type"]:
            problems.append(
                f"[{case_id}] note {idx}: directive_type mismatch (got {act['directive_type']}, expected {exp['directive_type']})"
            )
            continue
        if exp["directive_type"] == "no_op":
            continue

        exp_adj = exp["structured_adjustment"] or {}
        act_adj = act.get("structured_adjustment") or {}

        if act_adj.get("hours") != exp_adj.get("hours"):
            problems.append(
                f"[{case_id}] note {idx}: hours mismatch (got {act_adj.get('hours')}, expected {exp_adj.get('hours')})"
            )

        for numeric_field in ("factor", "minimum_energy_kwh", "max_grid_kwh"):
            if numeric_field in exp_adj:
                got_value = act_adj.get(numeric_field)
                if got_value is None or not close(float(got_value), float(exp_adj[numeric_field])):
                    problems.append(
                        f"[{case_id}] note {idx}: {numeric_field} mismatch (got {got_value}, expected {exp_adj[numeric_field]})"
                    )

    return problems


def derive_constraints(hours_input: List[dict], battery: dict, directives: List[dict]):
    base_solar = {h["hour"]: h["solar_kwh"] for h in hours_input}
    solar_factor = {h: 1.0 for h in range(24)}
    min_reserve = {h: battery["minimum_energy_kwh"] for h in range(24)}
    no_charge = set()
    no_discharge = set()
    max_grid = {}

    for d in directives:
        if not d["applies"] or d["directive_type"] == "no_op":
            continue
        adj = d.get("structured_adjustment") or {}
        affected = adj.get("hours", [])
        if d["directive_type"] == "solar_reduction":
            for h in affected:
                solar_factor[h] *= float(adj["factor"])
        elif d["directive_type"] == "minimum_battery_reserve":
            for h in affected:
                min_reserve[h] = max(min_reserve[h], float(adj["minimum_energy_kwh"]))
        elif d["directive_type"] == "no_charge_window":
            no_charge.update(affected)
        elif d["directive_type"] == "no_discharge_window":
            no_discharge.update(affected)
        elif d["directive_type"] == "max_grid_window":
            for h in affected:
                max_grid[h] = min(max_grid.get(h, adj["max_grid_kwh"]), float(adj["max_grid_kwh"]))

    effective_solar = {h: base_solar[h] * solar_factor[h] for h in range(24)}
    return effective_solar, min_reserve, no_charge, no_discharge, max_grid


def replay_plan(case_id: str, hours_input: List[dict], battery: dict, directives: List[dict], plan: List[dict]) -> List[str]:
    problems = []
    demand = {h["hour"]: h["demand_kwh"] for h in hours_input}
    tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in hours_input}
    effective_solar, min_reserve, no_charge, no_discharge, max_grid = derive_constraints(
        hours_input, battery, directives
    )

    plan_by_hour = {p["hour"]: p for p in plan}
    if sorted(plan_by_hour.keys()) != list(range(24)):
        problems.append(f"[{case_id}] hourly_plan must contain exactly hours 0..23")
        return problems

    prev_energy = battery["initial_energy_kwh"]
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h in range(24):
        p = plan_by_hour[h]
        grid_kwh = p["grid_kwh"]
        solar_used = p["solar_used_kwh"]
        action = p["battery_action"]
        battery_kwh = p["battery_kwh"]
        energy_after = p["battery_energy_after_kwh"]

        charge = battery_kwh if action == "charge" else 0.0
        discharge = battery_kwh if action == "discharge" else 0.0

        if action not in ("charge", "discharge", "idle"):
            problems.append(f"[{case_id}] hour {h}: invalid battery_action {action}")
        if action == "idle" and not close(battery_kwh, 0.0):
            problems.append(f"[{case_id}] hour {h}: idle hour must have battery_kwh 0")

        if grid_kwh < -TOLERANCE:
            problems.append(f"[{case_id}] hour {h}: negative grid_kwh")
        if solar_used < -TOLERANCE or solar_used > effective_solar[h] + TOLERANCE:
            problems.append(
                f"[{case_id}] hour {h}: solar_used_kwh {solar_used} exceeds effective solar {effective_solar[h]}"
            )
        if not close(grid_kwh + solar_used + discharge, demand[h] + charge, tol=TOLERANCE * 2):
            problems.append(f"[{case_id}] hour {h}: energy balance violated")
        if not close(prev_energy + charge - discharge, energy_after, tol=TOLERANCE * 2):
            problems.append(f"[{case_id}] hour {h}: battery transition inconsistent")
        if energy_after < min_reserve[h] - TOLERANCE or energy_after > battery["capacity_kwh"] + TOLERANCE:
            problems.append(f"[{case_id}] hour {h}: battery energy out of bounds")
        if charge > battery["max_charge_kwh_per_hour"] + TOLERANCE:
            problems.append(f"[{case_id}] hour {h}: charge rate limit exceeded")
        if discharge > battery["max_discharge_kwh_per_hour"] + TOLERANCE:
            problems.append(f"[{case_id}] hour {h}: discharge rate limit exceeded")
        if h in no_charge and charge > TOLERANCE:
            problems.append(f"[{case_id}] hour {h}: no_charge_window violated")
        if h in no_discharge and discharge > TOLERANCE:
            problems.append(f"[{case_id}] hour {h}: no_discharge_window violated")
        if h in max_grid and grid_kwh > max_grid[h] + TOLERANCE:
            problems.append(f"[{case_id}] hour {h}: max_grid_window violated")

        prev_energy = energy_after
        total_grid += grid_kwh
        total_cost += grid_kwh * tariff[h]
        peak_grid = max(peak_grid, grid_kwh)

    if not close(prev_energy, battery["initial_energy_kwh"], tol=TOLERANCE * 2):
        problems.append(f"[{case_id}] end-of-day battery neutrality violated")

    return problems, total_grid, total_cost, peak_grid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument(
        "--samples",
        default=str(Path(__file__).resolve().parents[2] / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"),
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.samples))
    overall_ok = True

    for case in cases:
        case_id = case["id"]
        response = requests.post(f"{args.base_url}/optimize-energy", json=case["input"], timeout=30)
        if response.status_code != 200:
            print(f"[{case_id}] FAIL: HTTP {response.status_code}: {response.text[:300]}")
            overall_ok = False
            continue

        body = response.json()
        problems = []

        if body.get("scenario_id") != case["input"]["scenario_id"]:
            problems.append(f"[{case_id}] scenario_id mismatch")

        problems += check_interpretation(
            case_id, body.get("directive_interpretation", []), case["expected_output"]["directive_interpretation"]
        )

        replay_result = replay_plan(
            case_id,
            case["input"]["hours"],
            case["input"]["battery"],
            body.get("directive_interpretation", []),
            body.get("hourly_plan", []),
        )
        if isinstance(replay_result, tuple) and len(replay_result) == 4:
            replay_problems, total_grid, total_cost, peak_grid = replay_result
            problems += replay_problems
            if not close(total_grid, body.get("total_grid_kwh", -1), tol=0.1):
                problems.append(f"[{case_id}] total_grid_kwh does not match hourly_plan")
            if not close(total_cost, body.get("total_cost_bdt", -1), tol=0.1):
                problems.append(f"[{case_id}] total_cost_bdt does not match hourly_plan")
            if not close(peak_grid, body.get("peak_grid_kwh", -1), tol=0.1):
                problems.append(f"[{case_id}] peak_grid_kwh does not match hourly_plan")

            reference_cost = case["expected_output"]["total_cost_bdt"]
            quality_ratio = min(1.0, reference_cost / total_cost) if total_cost > 0 else 1.0
            print(
                f"[{case_id}] cost={total_cost:.2f} BDT vs reference={reference_cost:.2f} BDT "
                f"(quality_ratio={quality_ratio:.3f})"
            )
        else:
            problems += replay_result

        if problems:
            overall_ok = False
            print(f"[{case_id}] FAIL:")
            for p in problems:
                print(f"    - {p}")
        else:
            print(f"[{case_id}] PASS")

    print("\nOVERALL:", "PASS" if overall_ok else "FAIL")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
