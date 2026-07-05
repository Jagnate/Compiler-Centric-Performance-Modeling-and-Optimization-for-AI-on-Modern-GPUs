from __future__ import annotations

import csv
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from .models import (
    AggregateUtilization,
    FastCohortResult,
    GemmModel,
    HardwareConfig,
    OccupancyLimits,
    ResourceVectorUs,
    ScheduleEventResult,
    SimulationResult,
)
from .occupancy import estimate_occupancy
from .tile_cache import TileLRU


CtaAssignment = tuple[int, int, int]
SimulationMode = Literal["fast", "detailed"]
RESOURCE_EPSILON = 1e-6


@dataclass
class ResourceWork:
    tc: float = 0.0
    smem: float = 0.0
    l2: float = 0.0
    ddr: float = 0.0

    def complete(self) -> bool:
        return all(
            value <= RESOURCE_EPSILON
            for value in (self.tc, self.smem, self.l2, self.ddr)
        )


@dataclass
class RunningBatch:
    event_index: int
    sm_id: int
    start_us: float
    batch: list[CtaAssignment]
    active_sms_at_start: int
    active_ctas_at_start: int
    effective_depth: int
    steady_iters: int
    l2_hit_rate: float
    load_l2_bytes: int
    load_ddr_bytes: int
    store_ddr_bytes: int
    load_vector_us_per_k: ResourceVectorUs
    compute_vector_us_per_k: ResourceVectorUs
    store_vector_us: ResourceVectorUs
    steady_bottleneck: str
    phases: list[ResourceWork]
    phase_index: int
    phase_elapsed_us: list[float]
    dynamic_rate_updates: int = 0

    @property
    def ctas(self) -> int:
        return len(self.batch)

    def current_phase(self) -> ResourceWork:
        return self.phases[self.phase_index]

    def advance_empty_phases(self) -> None:
        while self.phase_index < len(self.phases) and self.phases[self.phase_index].complete():
            self.phase_index += 1

    def complete(self) -> bool:
        return self.phase_index >= len(self.phases)


def cta_launch_order(model: GemmModel) -> list[CtaAssignment]:
    """Return logical CTA index and row-major output-tile coordinates."""
    return [
        (by * model.grid_x + bx, by, bx)
        for by in range(model.grid_y)
        for bx in range(model.grid_x)
    ]


def per_cta_rates(
    hw: HardwareConfig,
    *,
    active_sms: int,
    active_ctas: int,
    resident_ctas_on_sm: int,
) -> dict[str, float]:
    """Allocate local SM resources and global memory resources to one CTA."""
    if active_sms <= 0 or active_ctas <= 0 or resident_ctas_on_sm <= 0:
        raise ValueError("Active SM and CTA counts must be positive.")

    local_divisor = hw.num_sms * resident_ctas_on_sm
    global_divisor = active_ctas
    return {
        "tc_flops_s": hw.tensor_peak_tflops * hw.tensor_efficiency * 1e12 / local_divisor,
        "smem_bytes_s": hw.smem_bandwidth_gbs * hw.smem_efficiency * 1e9 / local_divisor,
        "l2_bytes_s": hw.l2_bandwidth_gbs * hw.l2_efficiency * 1e9 / global_divisor,
        "ddr_bytes_s": hw.ddr_bandwidth_gbs * hw.ddr_efficiency * 1e9 / global_divisor,
    }


def bottleneck_name(v: ResourceVectorUs) -> str:
    values = {"tc": v.tc, "smem": v.smem, "l2": v.l2, "ddr": v.ddr}
    return max(values, key=values.get)


def pipeline_envelope_us(
    *,
    load_vec: ResourceVectorUs,
    compute_vec: ResourceVectorUs,
    store_vec: ResourceVectorUs,
    k_tiles: int,
    pipeline_stages: int,
    resident_ctas_on_sm: int,
) -> tuple[int, float, float, float, float, float]:
    """Evaluate one resident CTA batch with a fixed-rate pipeline envelope."""
    effective_depth = max(0, pipeline_stages * resident_ctas_on_sm - 1)
    fill_iters = min(effective_depth, k_tiles)
    steady_iters = max(k_tiles - effective_depth, 0)
    steady_vec = load_vec.add(compute_vec)
    t_pro = fill_iters * load_vec.steady_time()
    t_steady = steady_vec.steady_time()
    t_epi = fill_iters * compute_vec.steady_time()
    t_store = store_vec.steady_time()
    latency = t_pro + steady_iters * t_steady + t_epi + t_store
    return effective_depth, t_pro, t_steady, t_epi, t_store, latency


def make_resource_vectors(
    *,
    model: GemmModel,
    hw: HardwareConfig,
    active_sms: int,
    active_ctas: int,
    resident_ctas_on_sm: int,
    l2_bytes_per_k_per_cta: float,
    ddr_bytes_per_k_per_cta: float,
) -> tuple[ResourceVectorUs, ResourceVectorUs, ResourceVectorUs]:
    rates = per_cta_rates(
        hw,
        active_sms=active_sms,
        active_ctas=active_ctas,
        resident_ctas_on_sm=resident_ctas_on_sm,
    )
    load_bytes_per_k = (
        model.block_M * model.block_K * model.bytes_a_per_element
        + model.block_K * model.block_N * model.bytes_b_per_element
    )
    compute_flops_per_k = 2 * model.block_M * model.block_N * model.block_K
    store_bytes = model.block_M * model.block_N * model.bytes_c_per_element
    return (
        ResourceVectorUs(
            smem=load_bytes_per_k / rates["smem_bytes_s"] * 1e6,
            l2=l2_bytes_per_k_per_cta / rates["l2_bytes_s"] * 1e6,
            ddr=ddr_bytes_per_k_per_cta / rates["ddr_bytes_s"] * 1e6,
        ),
        ResourceVectorUs(tc=compute_flops_per_k / rates["tc_flops_s"] * 1e6),
        ResourceVectorUs(
            l2=store_bytes / rates["l2_bytes_s"] * 1e6,
            ddr=store_bytes / rates["ddr_bytes_s"] * 1e6,
        ),
    )


def aggregate_fast_cache_traffic(model: GemmModel) -> tuple[int, int, float]:
    """Approximate GEMM cache traffic from unique A/B tiles without tracing CTAs."""
    a_tile_bytes = model.block_M * model.block_K * model.bytes_a_per_element
    b_tile_bytes = model.block_K * model.block_N * model.bytes_b_per_element
    total_l2 = model.num_ctas * model.k_tiles * (a_tile_bytes + b_tile_bytes)
    unique_a = model.grid_y * model.k_tiles * a_tile_bytes
    unique_b = model.grid_x * model.k_tiles * b_tile_bytes
    total_ddr = min(total_l2, unique_a + unique_b)
    hit_rate = 1.0 - total_ddr / total_l2 if total_l2 else 0.0
    return total_l2, total_ddr, hit_rate


def fast_cohort_latency(
    *,
    wave_kind: str,
    wave_multiplicity: int,
    sm_count: int,
    active_ctas: int,
    ctas_per_sm: int,
    l2_hit_rate: float,
    model: GemmModel,
    hw: HardwareConfig,
) -> FastCohortResult:
    load_bytes_per_k = (
        model.block_M * model.block_K * model.bytes_a_per_element
        + model.block_K * model.block_N * model.bytes_b_per_element
    )
    load_vec, compute_vec, store_vec = make_resource_vectors(
        model=model,
        hw=hw,
        active_sms=sm_count,
        active_ctas=active_ctas,
        resident_ctas_on_sm=ctas_per_sm,
        l2_bytes_per_k_per_cta=load_bytes_per_k,
        ddr_bytes_per_k_per_cta=load_bytes_per_k * (1.0 - l2_hit_rate),
    )
    effective_depth, _pro, _steady, _epi, _store, latency = pipeline_envelope_us(
        load_vec=load_vec,
        compute_vec=compute_vec,
        store_vec=store_vec,
        k_tiles=model.k_tiles,
        pipeline_stages=model.pipeline_stages or 1,
        resident_ctas_on_sm=ctas_per_sm,
    )
    return FastCohortResult(
        wave_kind=wave_kind,
        wave_multiplicity=wave_multiplicity,
        sm_count=sm_count,
        active_ctas=active_ctas,
        ctas_per_sm=ctas_per_sm,
        effective_depth=effective_depth,
        latency_us=latency,
        steady_bottleneck=bottleneck_name(load_vec.add(compute_vec)),
    )


def simulate_fast(
    model: GemmModel,
    hw: HardwareConfig,
    *,
    resident_capacity: int,
    occupancy: OccupancyLimits,
) -> SimulationResult:
    """Evaluate full waves and tail residency cohorts without per-CTA events."""
    full_capacity = hw.num_sms * resident_capacity
    full_waves = model.num_ctas // full_capacity
    tail_ctas = model.num_ctas % full_capacity
    total_l2, total_ddr, hit_rate = aggregate_fast_cache_traffic(model)
    cohorts: list[FastCohortResult] = []
    total_latency_us = 0.0

    if full_waves:
        full = fast_cohort_latency(
            wave_kind="full",
            wave_multiplicity=full_waves,
            sm_count=hw.num_sms,
            active_ctas=full_capacity,
            ctas_per_sm=resident_capacity,
            l2_hit_rate=hit_rate,
            model=model,
            hw=hw,
        )
        cohorts.append(full)
        total_latency_us += full_waves * full.latency_us

    if tail_ctas:
        active_sms = min(tail_ctas, hw.num_sms)
        base = tail_ctas // active_sms
        extra = tail_ctas % active_sms
        tail_cohorts: list[FastCohortResult] = []
        if active_sms - extra:
            tail_cohorts.append(
                fast_cohort_latency(
                    wave_kind="tail",
                    wave_multiplicity=1,
                    sm_count=active_sms - extra,
                    active_ctas=tail_ctas,
                    ctas_per_sm=base,
                    l2_hit_rate=hit_rate,
                    model=model,
                    hw=hw,
                )
            )
        if extra:
            tail_cohorts.append(
                fast_cohort_latency(
                    wave_kind="tail",
                    wave_multiplicity=1,
                    sm_count=extra,
                    active_ctas=tail_ctas,
                    ctas_per_sm=base + 1,
                    l2_hit_rate=hit_rate,
                    model=model,
                    hw=hw,
                )
            )
        cohorts.extend(tail_cohorts)
        total_latency_us += max(cohort.latency_us for cohort in tail_cohorts)

    store_bytes = model.num_ctas * model.block_M * model.block_N * model.bytes_c_per_element
    sm_active_time_us = sum(
        cohort.wave_multiplicity * cohort.sm_count * cohort.latency_us
        for cohort in cohorts
    )
    resident_cta_time_us = sum(
        cohort.wave_multiplicity
        * cohort.sm_count
        * cohort.ctas_per_sm
        * cohort.latency_us
        for cohort in cohorts
    )
    return make_simulation_result(
        model=model,
        hw=hw,
        occupancy=occupancy,
        total_latency_us=total_latency_us,
        overall_l2_hit=hit_rate,
        full_capacity=full_capacity,
        full_waves=full_waves,
        tail_ctas=tail_ctas,
        schedule_model="analytical_wave_cohorts_v1",
        fast_cohorts=cohorts,
        schedule_events=[],
        total_l2_bytes=total_l2 + store_bytes,
        total_ddr_bytes=total_ddr + store_bytes,
        sm_active_time_us=sm_active_time_us,
        resident_cta_time_us=resident_cta_time_us,
    )


def classify_batch_traffic(
    batches: dict[int, list[CtaAssignment]],
    model: GemmModel,
    lru: TileLRU,
) -> dict[int, tuple[int, int, int]]:
    a_tile_bytes = model.block_M * model.block_K * model.bytes_a_per_element
    b_tile_bytes = model.block_K * model.block_N * model.bytes_b_per_element
    traffic = {sm_id: [0, 0, 0] for sm_id in batches}
    assignments = sorted(
        (assignment, sm_id)
        for sm_id, batch in batches.items()
        for assignment in batch
    )
    for (_cta_index, by, bx), sm_id in assignments:
        for k in range(model.k_tiles):
            for key, size in ((("A", by, k), a_tile_bytes), (("B", k, bx), b_tile_bytes)):
                traffic[sm_id][0] += size
                if lru.access(key, size):
                    traffic[sm_id][2] += size
                else:
                    traffic[sm_id][1] += size
    return {sm_id: tuple(values) for sm_id, values in traffic.items()}


def build_running_batch(
    *,
    event_index: int,
    sm_id: int,
    start_us: float,
    batch: list[CtaAssignment],
    active_sms: int,
    active_ctas: int,
    traffic: tuple[int, int, int],
    model: GemmModel,
    hw: HardwareConfig,
) -> RunningBatch:
    resident_ctas = len(batch)
    l2_bytes, ddr_bytes, hit_bytes = traffic
    load_bytes_per_k = (
        model.block_M * model.block_K * model.bytes_a_per_element
        + model.block_K * model.block_N * model.bytes_b_per_element
    )
    compute_flops_per_k = 2 * model.block_M * model.block_N * model.block_K
    store_bytes_per_cta = model.block_M * model.block_N * model.bytes_c_per_element
    total_load_bytes = resident_ctas * model.k_tiles * load_bytes_per_k
    hit_rate = hit_bytes / total_load_bytes if total_load_bytes else 0.0
    l2_per_k_per_cta = l2_bytes / resident_ctas / model.k_tiles
    ddr_per_k_per_cta = ddr_bytes / resident_ctas / model.k_tiles
    load_vec, compute_vec, store_vec = make_resource_vectors(
        model=model,
        hw=hw,
        active_sms=active_sms,
        active_ctas=active_ctas,
        resident_ctas_on_sm=resident_ctas,
        l2_bytes_per_k_per_cta=l2_per_k_per_cta,
        ddr_bytes_per_k_per_cta=ddr_per_k_per_cta,
    )

    effective_depth = max(0, (model.pipeline_stages or 1) * resident_ctas - 1)
    fill_iters = min(effective_depth, model.k_tiles)
    steady_iters = max(model.k_tiles - effective_depth, 0)
    l2_per_iteration = l2_bytes / model.k_tiles
    ddr_per_iteration = ddr_bytes / model.k_tiles
    load_per_iteration = resident_ctas * load_bytes_per_k
    compute_per_iteration = resident_ctas * compute_flops_per_k
    phases = [
        ResourceWork(
            smem=fill_iters * load_per_iteration,
            l2=fill_iters * l2_per_iteration,
            ddr=fill_iters * ddr_per_iteration,
        ),
        ResourceWork(
            tc=steady_iters * compute_per_iteration,
            smem=steady_iters * load_per_iteration,
            l2=steady_iters * l2_per_iteration,
            ddr=steady_iters * ddr_per_iteration,
        ),
        ResourceWork(tc=fill_iters * compute_per_iteration),
        ResourceWork(
            l2=resident_ctas * store_bytes_per_cta,
            ddr=resident_ctas * store_bytes_per_cta,
        ),
    ]
    job = RunningBatch(
        event_index=event_index,
        sm_id=sm_id,
        start_us=start_us,
        batch=batch,
        active_sms_at_start=active_sms,
        active_ctas_at_start=active_ctas,
        effective_depth=effective_depth,
        steady_iters=steady_iters,
        l2_hit_rate=hit_rate,
        load_l2_bytes=l2_bytes,
        load_ddr_bytes=ddr_bytes,
        store_ddr_bytes=resident_ctas * store_bytes_per_cta,
        load_vector_us_per_k=load_vec,
        compute_vector_us_per_k=compute_vec,
        store_vector_us=store_vec,
        steady_bottleneck=bottleneck_name(load_vec.add(compute_vec)),
        phases=phases,
        phase_index=0,
        phase_elapsed_us=[0.0, 0.0, 0.0, 0.0],
    )
    job.advance_empty_phases()
    return job


def take_spread_first_batches(
    pending: deque[CtaAssignment],
    sm_ids: list[int],
    resident_capacity: int,
) -> dict[int, list[CtaAssignment]]:
    batches = {sm_id: [] for sm_id in sm_ids}
    for _layer in range(resident_capacity):
        for sm_id in sm_ids:
            if not pending:
                break
            batches[sm_id].append(pending.popleft())
        if not pending:
            break
    return {sm_id: batch for sm_id, batch in batches.items() if batch}


def dynamic_rates(
    running: dict[int, RunningBatch],
    hw: HardwareConfig,
) -> dict[int, ResourceWork]:
    l2_weight = sum(
        job.ctas for job in running.values() if job.current_phase().l2 > RESOURCE_EPSILON
    )
    ddr_weight = sum(
        job.ctas for job in running.values() if job.current_phase().ddr > RESOURCE_EPSILON
    )
    tc_per_sm_us = hw.tensor_peak_tflops * hw.tensor_efficiency * 1e12 / hw.num_sms / 1e6
    smem_per_sm_us = hw.smem_bandwidth_gbs * hw.smem_efficiency * 1e9 / hw.num_sms / 1e6
    l2_total_us = hw.l2_bandwidth_gbs * hw.l2_efficiency * 1e9 / 1e6
    ddr_total_us = hw.ddr_bandwidth_gbs * hw.ddr_efficiency * 1e9 / 1e6

    rates: dict[int, ResourceWork] = {}
    for sm_id, job in running.items():
        phase = job.current_phase()
        rates[sm_id] = ResourceWork(
            tc=tc_per_sm_us if phase.tc > RESOURCE_EPSILON else 0.0,
            smem=smem_per_sm_us if phase.smem > RESOURCE_EPSILON else 0.0,
            l2=(l2_total_us * job.ctas / l2_weight)
            if phase.l2 > RESOURCE_EPSILON and l2_weight
            else 0.0,
            ddr=(ddr_total_us * job.ctas / ddr_weight)
            if phase.ddr > RESOURCE_EPSILON and ddr_weight
            else 0.0,
        )
    return rates


def next_dynamic_interval(
    running: dict[int, RunningBatch],
    rates: dict[int, ResourceWork],
) -> float:
    intervals: list[float] = []
    for sm_id, job in running.items():
        phase = job.current_phase()
        rate = rates[sm_id]
        for remaining, resource_rate in (
            (phase.tc, rate.tc),
            (phase.smem, rate.smem),
            (phase.l2, rate.l2),
            (phase.ddr, rate.ddr),
        ):
            if remaining > RESOURCE_EPSILON:
                if resource_rate <= 0:
                    raise ValueError("Positive resource work has no allocated rate.")
                intervals.append(remaining / resource_rate)
    if not intervals:
        raise ValueError("Dynamic scheduler has running jobs but no remaining work.")
    return min(intervals)


def advance_dynamic_work(
    running: dict[int, RunningBatch],
    rates: dict[int, ResourceWork],
    elapsed_us: float,
) -> None:
    for sm_id, job in running.items():
        phase = job.current_phase()
        rate = rates[sm_id]
        phase.tc = max(0.0, phase.tc - rate.tc * elapsed_us)
        phase.smem = max(0.0, phase.smem - rate.smem * elapsed_us)
        phase.l2 = max(0.0, phase.l2 - rate.l2 * elapsed_us)
        phase.ddr = max(0.0, phase.ddr - rate.ddr * elapsed_us)
        job.phase_elapsed_us[job.phase_index] += elapsed_us
        job.dynamic_rate_updates += 1


def finalize_event(job: RunningBatch, end_us: float) -> ScheduleEventResult:
    latency = end_us - job.start_us
    steady_per_iteration = (
        job.phase_elapsed_us[1] / job.steady_iters if job.steady_iters else 0.0
    )
    return ScheduleEventResult(
        event_index=job.event_index,
        sm_id=job.sm_id,
        start_us=job.start_us,
        end_us=end_us,
        ctas=job.ctas,
        cta_indices=[assignment[0] for assignment in job.batch],
        active_sms_at_start=job.active_sms_at_start,
        active_ctas_at_start=job.active_ctas_at_start,
        resident_ctas_on_sm=job.ctas,
        effective_depth=job.effective_depth,
        l2_hit_rate=job.l2_hit_rate,
        load_l2_bytes=job.load_l2_bytes,
        load_ddr_bytes=job.load_ddr_bytes,
        store_ddr_bytes=job.store_ddr_bytes,
        load_vector_us_per_k=job.load_vector_us_per_k,
        compute_vector_us_per_k=job.compute_vector_us_per_k,
        store_vector_us=job.store_vector_us,
        t_pro_us=job.phase_elapsed_us[0],
        t_steady_us=steady_per_iteration,
        t_epi_us=job.phase_elapsed_us[2],
        t_store_us=job.phase_elapsed_us[3],
        latency_us=latency,
        steady_bottleneck=job.steady_bottleneck,
        dynamic_rate_updates=job.dynamic_rate_updates,
    )


def simulate_detailed(
    model: GemmModel,
    hw: HardwareConfig,
    *,
    resident_capacity: int,
    occupancy: OccupancyLimits,
) -> SimulationResult:
    pending = deque(cta_launch_order(model))
    lru = TileLRU(hw.l2_capacity_bytes)
    running: dict[int, RunningBatch] = {}
    completed_events: list[ScheduleEventResult] = []
    next_event_index = 0
    current_time_us = 0.0

    def dispatch(sm_ids: list[int]) -> None:
        nonlocal next_event_index
        new_batches = take_spread_first_batches(pending, sorted(sm_ids), resident_capacity)
        if not new_batches:
            return
        active_sms = len(running) + len(new_batches)
        active_ctas = sum(job.ctas for job in running.values()) + sum(
            len(batch) for batch in new_batches.values()
        )
        traffic_by_sm = classify_batch_traffic(new_batches, model, lru)
        for sm_id in sorted(new_batches):
            running[sm_id] = build_running_batch(
                event_index=next_event_index,
                sm_id=sm_id,
                start_us=current_time_us,
                batch=new_batches[sm_id],
                active_sms=active_sms,
                active_ctas=active_ctas,
                traffic=traffic_by_sm[sm_id],
                model=model,
                hw=hw,
            )
            next_event_index += 1

    dispatch(list(range(hw.num_sms)))
    while running:
        rates = dynamic_rates(running, hw)
        elapsed_us = next_dynamic_interval(running, rates)
        advance_dynamic_work(running, rates, elapsed_us)
        current_time_us += elapsed_us

        completed_sms: list[int] = []
        for sm_id, job in list(running.items()):
            job.advance_empty_phases()
            if job.complete():
                completed_events.append(finalize_event(job, current_time_us))
                completed_sms.append(sm_id)
                del running[sm_id]
        if pending and completed_sms:
            dispatch(completed_sms)

    completed_events.sort(key=lambda event: event.event_index)
    full_capacity = hw.num_sms * resident_capacity
    full_waves = model.num_ctas // full_capacity
    tail_ctas = model.num_ctas % full_capacity
    total_l2 = sum(event.load_l2_bytes for event in completed_events)
    total_ddr = sum(event.load_ddr_bytes for event in completed_events)
    total_store = sum(event.store_ddr_bytes for event in completed_events)
    hit_rate = 1.0 - total_ddr / total_l2 if total_l2 else 0.0
    return make_simulation_result(
        model=model,
        hw=hw,
        occupancy=occupancy,
        total_latency_us=current_time_us,
        overall_l2_hit=hit_rate,
        full_capacity=full_capacity,
        full_waves=full_waves,
        tail_ctas=tail_ctas,
        schedule_model="dynamic_rolling_sm_batches_v1",
        fast_cohorts=[],
        schedule_events=completed_events,
        total_l2_bytes=total_l2 + total_store,
        total_ddr_bytes=total_ddr + total_store,
        sm_active_time_us=sum(event.latency_us for event in completed_events),
        resident_cta_time_us=sum(
            event.ctas * event.latency_us for event in completed_events
        ),
    )


def make_simulation_result(
    *,
    model: GemmModel,
    hw: HardwareConfig,
    occupancy: OccupancyLimits,
    total_latency_us: float,
    overall_l2_hit: float,
    full_capacity: int,
    full_waves: int,
    tail_ctas: int,
    schedule_model: str,
    fast_cohorts: list[FastCohortResult],
    schedule_events: list[ScheduleEventResult],
    total_l2_bytes: int,
    total_ddr_bytes: int,
    sm_active_time_us: float,
    resident_cta_time_us: float,
) -> SimulationResult:
    if total_latency_us <= 0:
        raise ValueError("Simulation produced no positive latency.")
    total_latency_ms = total_latency_us / 1000.0
    latency_s = total_latency_us / 1e6
    load_bytes_per_k = (
        model.block_M * model.block_K * model.bytes_a_per_element
        + model.block_K * model.block_N * model.bytes_b_per_element
    )
    total_smem_bytes = model.num_ctas * model.k_tiles * load_bytes_per_k
    utilization = AggregateUtilization(
        capacity_basis="configured_physical_peak",
        sm_activity=sm_active_time_us / (hw.num_sms * total_latency_us),
        resident_cta_slot_utilization=resident_cta_time_us
        / (full_capacity * total_latency_us),
        theoretical_occupancy=occupancy.resident_ctas_per_sm
        * model.warps_per_cta
        / (hw.max_threads_per_sm / 32),
        achieved_occupancy=resident_cta_time_us
        * model.warps_per_cta
        / (hw.num_sms * (hw.max_threads_per_sm / 32) * total_latency_us),
        tensor_core_utilization=model.flops
        / (hw.tensor_peak_tflops * 1e12 * latency_s),
        smem_utilization=total_smem_bytes
        / (hw.smem_bandwidth_gbs * 1e9 * latency_s),
        l2_utilization=total_l2_bytes
        / (hw.l2_bandwidth_gbs * 1e9 * latency_s),
        hbm_utilization=total_ddr_bytes
        / (hw.ddr_bandwidth_gbs * 1e9 * latency_s),
    )
    return SimulationResult(
        model_name="tile_centric_gemm_v3",
        kernel_name=model.kernel_name,
        arch=model.arch,
        M=model.M,
        N=model.N,
        K=model.K,
        block_M=model.block_M,
        block_N=model.block_N,
        block_K=model.block_K,
        k_tiles=model.k_tiles,
        grid_x=model.grid_x,
        grid_y=model.grid_y,
        num_ctas=model.num_ctas,
        pipeline_stages=model.pipeline_stages or 1,
        resident_ctas_per_sm=occupancy.resident_ctas_per_sm,
        occupancy=occupancy,
        full_wave_capacity_ctas=full_capacity,
        full_waves=full_waves,
        tail_ctas=tail_ctas,
        total_latency_us=total_latency_us,
        total_latency_ms=total_latency_ms,
        estimated_tflops=model.flops / (total_latency_ms * 1e9),
        overall_l2_hit_rate=overall_l2_hit,
        hardware=hw,
        aggregate_utilization=utilization,
        schedule_model=schedule_model,
        fast_cohorts=fast_cohorts,
        schedule_events=schedule_events,
    )


def simulate(
    model: GemmModel,
    hw: HardwareConfig,
    mode: SimulationMode = "fast",
) -> SimulationResult:
    occupancy = estimate_occupancy(
        hw=hw,
        threads_per_cta=model.threads_per_cta,
        warps_per_cta=model.warps_per_cta,
        dynamic_shared_bytes_per_cta=model.dynamic_shared_bytes_per_cta,
        registers_per_thread=model.registers_per_thread,
    )
    if mode == "fast":
        return simulate_fast(
            model,
            hw,
            resident_capacity=occupancy.resident_ctas_per_sm,
            occupancy=occupancy,
        )
    if mode == "detailed":
        return simulate_detailed(
            model,
            hw,
            resident_capacity=occupancy.resident_ctas_per_sm,
            occupancy=occupancy,
        )
    raise ValueError(f"Unknown simulation mode: {mode!r}")


def write_schedule_csv(path: Path, events: list[ScheduleEventResult]) -> None:
    rows: list[dict[str, Any]] = []
    for event in events:
        row = asdict(event)
        for key in ("load_vector_us_per_k", "compute_vector_us_per_k", "store_vector_us"):
            vec = row.pop(key)
            for resource, value in vec.items():
                row[f"{key}_{resource}"] = value
        rows.append(row)
    if not rows:
        raise ValueError("Detailed schedule output requested without schedule events.")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
