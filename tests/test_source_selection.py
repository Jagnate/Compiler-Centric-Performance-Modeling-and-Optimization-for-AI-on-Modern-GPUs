"""Tests for trust-adaptive and source-diverse candidate promotion."""

from __future__ import annotations

import unittest

from kernel_optimization.schema import (
    BudgetConfig,
    Candidate,
    CandidateProposal,
    CandidateRecord,
    Measurement,
    ModelEvaluation,
    TaskSpec,
)
from kernel_optimization.selection import AdaptiveSelectionPolicy
from kernel_optimization.trust import TrustTracker


class SourceSelectionTests(unittest.TestCase):
    def test_extreme_model_error_temporarily_falls_back_to_measure_all(self) -> None:
        budget = BudgetConfig(
            min_promotions_per_round=2,
            max_promotions_per_round=4,
        )
        task = TaskSpec(
            task_id="trust-fallback-test",
            description="Fall back when model screening is unsupported.",
            reference="Return the input.",
            entrypoint="kernel",
            budget=budget,
        )
        seed = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        modeled = []
        for index in range(6):
            candidate = Candidate.from_proposal(
                task,
                seed,
                CandidateProposal(
                    hypothesis="Candidate %d." % index,
                    source_code=(
                        "def kernel(x):\n"
                        "    marker_%d = x\n"
                        "    return marker_%d\n" % (index, index)
                    ),
                ),
                generation=1,
            )
            modeled.append(
                CandidateRecord(
                    candidate=candidate,
                    model=ModelEvaluation(
                        valid=True,
                        predicted_latency_ms=float(index + 1),
                    ),
                )
            )
        trust = TrustTracker(
            relative_errors=[10.0, 9.0, 8.0],
            raw_relative_errors=[100.0, 90.0, 80.0],
        )

        selected = AdaptiveSelectionPolicy(budget).select(task, modeled, [], trust)

        self.assertEqual(len(selected), len(modeled))
        self.assertEqual(trust.screening_mode, "measure-all-fallback")
        self.assertTrue(
            all(
                any(
                    reason.startswith("trust-fallback-measure-all:")
                    for reason in record.selection_reasons
                )
                for record in selected
            )
        )

    def test_low_trust_increases_hardware_promotions(self) -> None:
        budget = BudgetConfig(
            min_promotions_per_round=2,
            max_promotions_per_round=5,
        )
        policy = AdaptiveSelectionPolicy(budget)
        low_trust = TrustTracker()
        high_trust = TrustTracker(
            relative_errors=[0.0, 0.0],
            direction_correct=2,
            direction_total=2,
        )

        self.assertEqual(policy.promotion_count(10, low_trust), 5)
        self.assertEqual(policy.promotion_count(10, high_trust), 2)

    def test_selection_includes_a_source_diversity_candidate(self) -> None:
        budget = BudgetConfig(
            min_promotions_per_round=4,
            max_promotions_per_round=4,
            random_seed=5,
        )
        task = TaskSpec(
            task_id="source-selection-test",
            description="Select diverse source candidates.",
            reference="Return the input.",
            entrypoint="kernel",
            budget=budget,
        )
        seed = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        seed_record = CandidateRecord(
            candidate=seed,
            model=ModelEvaluation(valid=True, predicted_latency_ms=10.0, confidence="high"),
            measurement=Measurement(correct=True, latency_ms=10.0),
        )
        sources = [
            "def kernel(x):\n    return x + 0\n",
            "def kernel(x):\n    y = x\n    return y\n",
            "def kernel(x):\n    return (x)\n",
            "def kernel(x):\n    for _ in range(1):\n        x = x\n    return x\n",
            "def kernel(x):\n    values = [x]\n    return values[0]\n",
        ]
        modeled = []
        for index, source in enumerate(sources, start=1):
            candidate = Candidate.from_proposal(
                task,
                seed,
                CandidateProposal(hypothesis="Candidate %d." % index, source_code=source),
                generation=1,
            )
            modeled.append(
                CandidateRecord(
                    candidate=candidate,
                    model=ModelEvaluation(
                        valid=True,
                        predicted_latency_ms=float(index),
                        confidence="low" if index == 3 else "high",
                    ),
                )
            )

        selected = AdaptiveSelectionPolicy(budget).select(
            task, modeled, [seed_record], TrustTracker()
        )
        reasons = {reason for record in selected for reason in record.selection_reasons}

        self.assertEqual(len(selected), 4)
        self.assertIn("model-top", reasons)
        self.assertIn("low-confidence-audit", reasons)
        self.assertIn("source-diversity", reasons)

    def test_selection_audits_a_strategy_lane_with_flat_model_scores(self) -> None:
        budget = BudgetConfig(
            min_promotions_per_round=3,
            max_promotions_per_round=3,
            random_seed=9,
        )
        task = TaskSpec(
            task_id="model-insensitive-selection-test",
            description="Audit candidates that the model cannot rank apart.",
            reference="Return the input.",
            entrypoint="kernel",
            budget=budget,
        )
        seed = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        beam = [
            CandidateRecord(
                candidate=seed,
                model=ModelEvaluation(valid=True, predicted_latency_ms=2.0),
                measurement=Measurement(correct=True, latency_ms=2.0),
            )
        ]
        modeled = []
        for index in range(5):
            candidate = Candidate.from_proposal(
                task,
                seed,
                CandidateProposal(
                    hypothesis="Flat model candidate %d." % index,
                    source_code=(
                        "def kernel(x):\n"
                        "    value_%d = x\n"
                        "    return value_%d\n" % (index, index)
                    ),
                    metadata={"strategy_id": "memory-layout"},
                ),
                generation=1,
            )
            modeled.append(
                CandidateRecord(
                    candidate=candidate,
                    model=ModelEvaluation(
                        valid=True,
                        predicted_latency_ms=1.0 + index * 0.0005,
                        confidence="high",
                    ),
                )
            )

        selected = AdaptiveSelectionPolicy(budget).select(
            task, modeled, beam, TrustTracker()
        )

        reasons = {
            reason for record in selected for reason in record.selection_reasons
        }
        self.assertIn("model-insensitive-audit", reasons)

    def test_selection_covers_distinct_strategy_lanes_before_random_audits(self) -> None:
        budget = BudgetConfig(
            min_promotions_per_round=3,
            max_promotions_per_round=3,
            random_seed=3,
        )
        task = TaskSpec(
            task_id="strategy-stratified-selection-test",
            description="Audit distinct structural directions.",
            reference="Return the input.",
            entrypoint="kernel",
            budget=budget,
        )
        seed = Candidate.seed(task, "def kernel(x):\n    return x\n", "kernel.py")
        modeled = []
        strategies = ("memory-layout", "memory-layout", "data-movement", "execution-mapping")
        for index, strategy in enumerate(strategies):
            candidate = Candidate.from_proposal(
                task,
                seed,
                CandidateProposal(
                    hypothesis="Try %s." % strategy,
                    source_code=(
                        "def kernel(x):\n"
                        "    value_%d = x\n"
                        "    return value_%d\n" % (index, index)
                    ),
                    metadata={"strategy_id": strategy},
                ),
                generation=1,
            )
            modeled.append(
                CandidateRecord(
                    candidate=candidate,
                    model=ModelEvaluation(
                        valid=True,
                        predicted_latency_ms=1.0 + index,
                        confidence="high",
                    ),
                )
            )

        selected = AdaptiveSelectionPolicy(budget).select(
            task,
            modeled,
            [],
            TrustTracker(relative_errors=[0.5, 0.5, 0.5]),
        )

        self.assertEqual(
            {record.candidate.proposal_metadata["strategy_id"] for record in selected},
            {"memory-layout", "data-movement", "execution-mapping"},
        )
        self.assertEqual(
            sum(
                "strategy-stratified-audit" in record.selection_reasons
                for record in selected
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
