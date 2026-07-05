from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from .models import PipelineNodeSummary


# Inspired by the paper's recursive prologue-steady-epilogue envelope:
# every loop is analyzed from child resource vectors before its parent composes it.
RESOURCE_EPSILON = 1e-6


@dataclass
class ResourceWork:
    """Work on independently schedulable hardware resource lanes."""

    tc: float = 0.0
    smem: float = 0.0
    l2: float = 0.0
    ddr: float = 0.0

    def add(self, other: "ResourceWork") -> "ResourceWork":
        return ResourceWork(
            tc=self.tc + other.tc,
            smem=self.smem + other.smem,
            l2=self.l2 + other.l2,
            ddr=self.ddr + other.ddr,
        )

    def scale(self, factor: int) -> "ResourceWork":
        return ResourceWork(
            tc=self.tc * factor,
            smem=self.smem * factor,
            l2=self.l2 * factor,
            ddr=self.ddr * factor,
        )

    def steady_time(self) -> float:
        return max(self.tc, self.smem, self.l2, self.ddr)

    def complete(self) -> bool:
        return all(
            value <= RESOURCE_EPSILON
            for value in (self.tc, self.smem, self.l2, self.ddr)
        )


@dataclass(frozen=True)
class ActionNode:
    name: str
    work: ResourceWork


@dataclass(frozen=True)
class SequenceNode:
    name: str
    children: tuple["PipelineNode", ...]


@dataclass(frozen=True)
class ConcurrentNode:
    name: str
    children: tuple["PipelineNode", ...]


@dataclass(frozen=True)
class PipelineLoopNode:
    name: str
    iterations: int
    pipeline_stages: int
    resident_tiles: int
    producer: "PipelineNode"
    consumer: "PipelineNode"


PipelineNode = Union[ActionNode, SequenceNode, ConcurrentNode, PipelineLoopNode]


@dataclass
class PipelinePhase:
    name: str
    work: ResourceWork
    minimum_duration: float = 0.0

    @property
    def duration(self) -> float:
        return max(self.work.steady_time(), self.minimum_duration)


@dataclass
class PipelineAnalysis:
    total_work: ResourceWork
    phases: list[PipelinePhase]
    summary: PipelineNodeSummary

    @property
    def latency(self) -> float:
        return sum(phase.duration for phase in self.phases)


def analyze_pipeline(
    node: PipelineNode,
    *,
    preserve_nested_latency: bool = True,
) -> PipelineAnalysis:
    """Recursively lower a pipeline tree into executable resource phases."""
    if isinstance(node, ActionNode):
        return PipelineAnalysis(
            total_work=node.work,
            phases=[PipelinePhase(name=node.name, work=node.work)],
            summary=PipelineNodeSummary(
                name=node.name,
                node_type="action",
            ),
        )

    if isinstance(node, SequenceNode):
        child_results = [
            analyze_pipeline(
                child,
                preserve_nested_latency=preserve_nested_latency,
            )
            for child in node.children
        ]
        total_work = ResourceWork()
        phases: list[PipelinePhase] = []
        for result in child_results:
            total_work = total_work.add(result.total_work)
            phases.extend(result.phases)
        return PipelineAnalysis(
            total_work=total_work,
            phases=phases,
            summary=PipelineNodeSummary(
                name=node.name,
                node_type="sequence",
                children=[result.summary for result in child_results],
            ),
        )

    if isinstance(node, ConcurrentNode):
        child_results = [
            analyze_pipeline(
                child,
                preserve_nested_latency=preserve_nested_latency,
            )
            for child in node.children
        ]
        total_work = ResourceWork()
        for result in child_results:
            total_work = total_work.add(result.total_work)
        return PipelineAnalysis(
            total_work=total_work,
            phases=[
                PipelinePhase(
                    name=node.name,
                    work=total_work,
                    minimum_duration=max(
                        (result.latency for result in child_results),
                        default=0.0,
                    )
                    if preserve_nested_latency
                    else 0.0,
                )
            ],
            summary=PipelineNodeSummary(
                name=node.name,
                node_type="concurrent",
                children=[result.summary for result in child_results],
            ),
        )

    if isinstance(node, PipelineLoopNode):
        if node.iterations <= 0:
            raise ValueError("Pipeline loop iterations must be positive.")
        if node.pipeline_stages <= 0 or node.resident_tiles <= 0:
            raise ValueError("Pipeline stages and resident tiles must be positive.")

        producer = analyze_pipeline(
            node.producer,
            preserve_nested_latency=preserve_nested_latency,
        )
        consumer = analyze_pipeline(
            node.consumer,
            preserve_nested_latency=preserve_nested_latency,
        )
        effective_depth = node.pipeline_stages * node.resident_tiles - 1
        fill_iterations = min(effective_depth, node.iterations)
        steady_iterations = max(node.iterations - effective_depth, 0)
        phases = [
            PipelinePhase(
                name=f"{node.name}.prologue",
                work=producer.total_work.scale(fill_iterations),
                minimum_duration=producer.latency * fill_iterations
                if preserve_nested_latency
                else 0.0,
            ),
            PipelinePhase(
                name=f"{node.name}.steady",
                work=producer.total_work.add(consumer.total_work).scale(
                    steady_iterations
                ),
                minimum_duration=max(producer.latency, consumer.latency)
                * steady_iterations
                if preserve_nested_latency
                else 0.0,
            ),
            PipelinePhase(
                name=f"{node.name}.epilogue",
                work=consumer.total_work.scale(fill_iterations),
                minimum_duration=consumer.latency * fill_iterations
                if preserve_nested_latency
                else 0.0,
            ),
        ]
        return PipelineAnalysis(
            total_work=producer.total_work.add(consumer.total_work).scale(
                node.iterations
            ),
            phases=phases,
            summary=PipelineNodeSummary(
                name=node.name,
                node_type="pipeline_loop",
                iterations=node.iterations,
                pipeline_stages=node.pipeline_stages,
                resident_tiles=node.resident_tiles,
                effective_depth=effective_depth,
                prologue_iterations=fill_iterations,
                steady_iterations=steady_iterations,
                epilogue_iterations=fill_iterations,
                children=[producer.summary, consumer.summary],
            ),
        )

    raise TypeError(f"Unsupported pipeline node: {type(node).__name__}")


def build_gemm_pipeline(
    *,
    k_tiles: int,
    pipeline_stages: int,
    resident_tiles: int,
    load_work: ResourceWork,
    compute_work: ResourceWork,
    store_work: ResourceWork,
    preserve_nested_latency: bool = True,
) -> PipelineAnalysis:
    """Build the recursive root -> K-loop -> action tree for one GEMM CTA batch."""
    root = SequenceNode(
        name="gemm_cta",
        children=(
            PipelineLoopNode(
                name="k_loop",
                iterations=k_tiles,
                pipeline_stages=pipeline_stages,
                resident_tiles=resident_tiles,
                producer=ActionNode(name="load_ab", work=load_work),
                consumer=ActionNode(name="mma_compute", work=compute_work),
            ),
            ActionNode(name="store_c", work=store_work),
        ),
    )
    return analyze_pipeline(
        root,
        preserve_nested_latency=preserve_nested_latency,
    )
