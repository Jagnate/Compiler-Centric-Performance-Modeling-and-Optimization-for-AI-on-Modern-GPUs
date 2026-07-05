from __future__ import annotations

import unittest

from perf_model.recursive_pipeline import (
    ActionNode,
    ConcurrentNode,
    PipelineLoopNode,
    ResourceWork,
    SequenceNode,
    analyze_pipeline,
    build_gemm_pipeline,
)


class RecursivePipelineTests(unittest.TestCase):
    def test_gemm_pipeline_expands_prologue_steady_epilogue(self) -> None:
        result = build_gemm_pipeline(
            k_tiles=8,
            pipeline_stages=3,
            resident_tiles=2,
            load_work=ResourceWork(smem=2.0, l2=3.0),
            compute_work=ResourceWork(tc=5.0),
            store_work=ResourceWork(ddr=7.0),
        )
        k_loop = result.summary.children[0]

        self.assertEqual(k_loop.effective_depth, 5)
        self.assertEqual(k_loop.prologue_iterations, 5)
        self.assertEqual(k_loop.steady_iterations, 3)
        self.assertEqual(k_loop.epilogue_iterations, 5)
        self.assertEqual(
            [phase.name for phase in result.phases],
            [
                "k_loop.prologue",
                "k_loop.steady",
                "k_loop.epilogue",
                "store_c",
            ],
        )
        self.assertAlmostEqual(result.latency, 62.0)

    def test_short_loop_has_no_negative_steady_state(self) -> None:
        result = build_gemm_pipeline(
            k_tiles=2,
            pipeline_stages=3,
            resident_tiles=2,
            load_work=ResourceWork(l2=3.0),
            compute_work=ResourceWork(tc=5.0),
            store_work=ResourceWork(ddr=7.0),
        )
        k_loop = result.summary.children[0]

        self.assertEqual(k_loop.effective_depth, 5)
        self.assertEqual(k_loop.steady_iterations, 0)
        self.assertAlmostEqual(result.latency, 23.0)

    def test_nested_pipeline_preserves_child_boundary_latency(self) -> None:
        inner = PipelineLoopNode(
            name="inner",
            iterations=4,
            pipeline_stages=2,
            resident_tiles=1,
            producer=ActionNode("inner_load", ResourceWork(smem=2.0)),
            consumer=ActionNode("inner_compute", ResourceWork(tc=3.0)),
        )
        outer = PipelineLoopNode(
            name="outer",
            iterations=3,
            pipeline_stages=2,
            resident_tiles=1,
            producer=inner,
            consumer=ActionNode("outer_compute", ResourceWork(tc=4.0)),
        )
        result = analyze_pipeline(outer)
        inner_summary = result.summary.children[0]

        self.assertEqual(inner_summary.node_type, "pipeline_loop")
        self.assertEqual(inner_summary.effective_depth, 1)
        # The inner envelope is 14 time units, while max(total resources) is 12.
        # The outer prologue must retain that two-unit boundary penalty.
        self.assertAlmostEqual(result.phases[0].minimum_duration, 14.0)
        self.assertGreater(result.latency, result.total_work.steady_time())

    def test_concurrent_children_share_independent_resource_lanes(self) -> None:
        root = SequenceNode(
            name="root",
            children=(
                ConcurrentNode(
                    name="load_pair",
                    children=(
                        ActionNode("load_a", ResourceWork(l2=3.0)),
                        ActionNode("load_b", ResourceWork(ddr=5.0)),
                    ),
                ),
                ActionNode("compute", ResourceWork(tc=7.0)),
            ),
        )
        result = analyze_pipeline(root)

        self.assertEqual(result.summary.children[0].node_type, "concurrent")
        self.assertAlmostEqual(result.phases[0].duration, 5.0)
        self.assertAlmostEqual(result.latency, 12.0)


if __name__ == "__main__":
    unittest.main()
