"""Deterministic parameter mutation for local end-to-end tests."""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List, Sequence, Set

from ..schema import Candidate, CandidateProposal, TaskSpec


class DeterministicMockGenerator:
    """Enumerate local one- and two-parameter edits without model access."""

    def generate(
        self,
        task: TaskSpec,
        parent: Candidate,
        evidence: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
        count: int,
    ) -> List[CandidateProposal]:
        del evidence
        seen = {
            self._parameter_key(dict(item.get("parameters") or {})) for item in history
        }
        edits = self._single_edits(task, parent)
        edits.extend(self._pair_edits(task, parent))

        proposals: List[CandidateProposal] = []
        emitted: Set[str] = set()
        for updates in edits:
            parameters = dict(parent.parameters)
            parameters.update(updates)
            key = self._parameter_key(parameters)
            if key in seen or key in emitted:
                continue
            emitted.add(key)
            changed = ", ".join(
                "%s=%r" % (name, value) for name, value in sorted(updates.items())
            )
            proposals.append(
                CandidateProposal(
                    hypothesis="Explore a local schedule mutation: %s." % changed,
                    parameter_updates=updates,
                    expected_effect={
                        "latency": "unknown",
                        "reason": "Deterministic search-space exploration",
                    },
                    metadata={"generator": "deterministic-mock"},
                )
            )
            if len(proposals) >= count:
                break
        return proposals

    @staticmethod
    def _single_edits(task: TaskSpec, parent: Candidate) -> List[Dict[str, Any]]:
        result = []
        for name, values in task.search_space.items():
            for value in values:
                if value != parent.parameters[name]:
                    result.append({name: value})
        return result

    @staticmethod
    def _pair_edits(task: TaskSpec, parent: Candidate) -> List[Dict[str, Any]]:
        alternatives = {
            name: [value for value in values if value != parent.parameters[name]]
            for name, values in task.search_space.items()
        }
        result = []
        for left, right in itertools.combinations(task.search_space, 2):
            for left_value in alternatives[left]:
                for right_value in alternatives[right]:
                    result.append({left: left_value, right: right_value})
        return result

    @staticmethod
    def _parameter_key(parameters: Dict[str, Any]) -> str:
        return json.dumps(parameters, sort_keys=True, separators=(",", ":"))

