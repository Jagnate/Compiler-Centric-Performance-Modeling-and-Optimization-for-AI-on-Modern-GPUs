"""Unit tests for candidate contracts and adaptive promotion."""

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


def make_task(budget: BudgetConfig | None = None) -> TaskSpec:
    return TaskSpec(
        task_id="selection-test",
        description="Exercise candidate validation and promotion.",
        reference="The output must match the reference implementation.",
        base_parameters={"tile": 0},
        search_space={"tile": tuple(range(6))},
        budget=budget or BudgetConfig(),
    )


class SchemaTests(unittest.TestCase):
    def test_rejects_candidate_outside_declared_search_space(self) -> None:
        task = make_task()
        parent = Candidate.seed(task)

        with self.assertRaisesRegex(ValueError, "outside the declared search space"):
            Candidate.from_proposal(
                task,
                parent,
                CandidateProposal(
                    hypothesis="Try an unsupported tile.",
                    parameter_updates={"tile": 99},
                ),
                generation=1,
            )

    def test_parameter_candidate_identity_is_independent_of_parent(self) -> None:
        task = make_task()
        seed = Candidate.seed(task)
        first_parent = Candidate.from_proposal(
            task,
            seed,
            CandidateProposal(
                hypothesis="Move to tile one.", parameter_updates={"tile": 1}
            ),
            generation=1,
        )
        from_seed = Candidate.from_proposal(
            task,
            seed,
            CandidateProposal(
                hypothesis="Move to tile two.", parameter_updates={"tile": 2}
            ),
            generation=1,
        )
        from_other_parent = Candidate.from_proposal(
            task,
            first_parent,
            CandidateProposal(
                hypothesis="Also reach tile two.", parameter_updates={"tile": 2}
            ),
            generation=2,
        )

        self.assertEqual(from_seed.candidate_id, from_other_parent.candidate_id)

    def test_source_patch_identity_retains_lineage(self) -> None:
        task = make_task()
        seed = Candidate.seed(task)
        other_parent = Candidate.from_proposal(
            task,
            seed,
            CandidateProposal(
                hypothesis="Move to tile one.", parameter_updates={"tile": 1}
            ),
            generation=1,
        )
        proposal = CandidateProposal(
            hypothesis="Apply a source-level schedule edit.",
            source_patch="@@ -1 +1 @@\n-old\n+new\n",
        )

        left = Candidate.from_proposal(task, seed, proposal, generation=1)
        right = Candidate.from_proposal(task, other_parent, proposal, generation=2)

        self.assertNotEqual(left.candidate_id, right.candidate_id)


class SelectionTests(unittest.TestCase):
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

    def test_selection_combines_exploitation_uncertainty_and_diversity(self) -> None:
        budget = BudgetConfig(
            min_promotions_per_round=4,
            max_promotions_per_round=4,
            random_seed=5,
        )
        task = make_task(budget)
        seed = Candidate.seed(task)
        seed_record = CandidateRecord(
            candidate=seed,
            model=ModelEvaluation(
                valid=True,
                predicted_latency_ms=10.0,
                confidence="high",
            ),
            measurement=Measurement(correct=True, latency_ms=10.0),
        )
        modeled = []
        for value in range(1, 6):
            candidate = Candidate.from_proposal(
                task,
                seed,
                CandidateProposal(
                    hypothesis="Try tile %d." % value,
                    parameter_updates={"tile": value},
                ),
                generation=1,
            )
            modeled.append(
                CandidateRecord(
                    candidate=candidate,
                    model=ModelEvaluation(
                        valid=True,
                        predicted_latency_ms=float(value),
                        confidence="low" if value == 3 else "high",
                    ),
                )
            )

        selected = AdaptiveSelectionPolicy(budget).select(
            task,
            modeled,
            [seed_record],
            TrustTracker(),
        )
        reasons = {
            reason for record in selected for reason in record.selection_reasons
        }

        self.assertEqual(len(selected), 4)
        self.assertIn("model-top", reasons)
        self.assertIn("low-confidence-audit", reasons)
        self.assertIn("parameter-diversity", reasons)


if __name__ == "__main__":
    unittest.main()

