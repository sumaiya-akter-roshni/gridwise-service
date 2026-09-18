"""FastAPI service exposing GET /health and POST /optimize-energy
(Problem Statement Section 06; Participant Guide Section 02/03).

End-to-end flow per Problem Statement Section 03:
Energy Data + Operator Notes -> LLM Interpreter -> Guardrail Validator -> Math Optimizer
-> Final Validator -> API Response.
"""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .directive_engine import apply_directives
from .guardrails import validate_and_normalize
from .llm_interpreter import interpret_operator_notes
from .optimizer import OptimizationInfeasibleError, solve_schedule
from .schemas import (
    BatteryConfig,
    DirectiveInterpretation,
    HealthResponse,
    OptimizeEnergyRequest,
    OptimizeEnergyResponse,
)

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("gridwise")

REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "25"))

app = FastAPI(title="GridWise Energy Optimization Service")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": "Malformed or structurally invalid request", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error while processing %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/optimize-energy", response_model=OptimizeEnergyResponse)
async def optimize_energy(payload: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_process_request, payload), timeout=REQUEST_TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        logger.error("Request timed out for scenario %s", payload.scenario_id)
        return JSONResponse(  # type: ignore[return-value]
            status_code=500, content={"detail": "Request exceeded internal timeout budget"}
        )


def _process_request(payload: OptimizeEnergyRequest) -> OptimizeEnergyResponse:
    raw_directives = _interpret_with_retry(payload)
    directives = validate_and_normalize(raw_directives, len(payload.operator_notes), payload.battery)

    effective_solar, min_reserve, no_charge_hours, no_discharge_hours, max_grid = apply_directives(
        payload.hours, payload.battery, directives
    )

    try:
        hourly_plan, total_grid_kwh, total_cost_bdt, peak_grid_kwh = solve_schedule(
            payload.hours,
            payload.battery,
            effective_solar,
            min_reserve,
            no_charge_hours,
            no_discharge_hours,
            max_grid,
        )
    except OptimizationInfeasibleError:
        logger.exception("Optimization infeasible for scenario %s", payload.scenario_id)
        raise

    plan_summary = _build_plan_summary(directives, total_cost_bdt, peak_grid_kwh)

    return OptimizeEnergyResponse(
        scenario_id=payload.scenario_id,
        directive_interpretation=directives,
        hourly_plan=hourly_plan,
        total_grid_kwh=total_grid_kwh,
        total_cost_bdt=total_cost_bdt,
        peak_grid_kwh=peak_grid_kwh,
        plan_summary=plan_summary,
    )


def _interpret_with_retry(payload: OptimizeEnergyRequest):
    """Calls the LLM, retrying once on transport/provider failure.

    If the LLM is unavailable after the retry, falls back to an empty directive list --
    validate_and_normalize() will safely turn that into no_op for every note rather than
    crashing the service (Problem Statement Section 08, "SAFE FAILURE").
    """
    for attempt in (1, 2):
        try:
            return interpret_operator_notes(payload.operator_notes, payload.battery)
        except Exception:
            logger.exception(
                "LLM interpretation attempt %d/2 failed for scenario %s", attempt, payload.scenario_id
            )
    return []


def _build_plan_summary(
    directives: list[DirectiveInterpretation], total_cost_bdt: float, peak_grid_kwh: float
) -> str:
    applied = [d for d in directives if d.applies]
    if not applied:
        return (
            f"No operator directives applied. Optimized 24-hour schedule costs "
            f"BDT {total_cost_bdt:.2f} with a peak hourly grid draw of {peak_grid_kwh:.2f} kWh."
        )
    kinds = ", ".join(sorted({d.directive_type for d in applied}))
    return (
        f"Applied {len(applied)} operator directive(s) ({kinds}). Optimized 24-hour "
        f"schedule costs BDT {total_cost_bdt:.2f} with a peak hourly grid draw of "
        f"{peak_grid_kwh:.2f} kWh."
    )
