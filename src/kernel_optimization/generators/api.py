"""Hosted OpenAI-compatible chat-completions candidate generator."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..prompts import SYSTEM_PROMPT, build_optimization_prompt
from ..schema import Candidate, CandidateProposal, TaskSpec


Transport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]
]


@dataclass(frozen=True)
class ApiGeneratorConfig:
    """Connection and sampling settings for a hosted chat-completions API."""

    api_url: str
    model: str
    api_key_environment_variable: str = "KERNEL_OPT_API_KEY"
    timeout_seconds: float = 120.0
    temperature: float = 0.4

    def __post_init__(self) -> None:
        if not self.api_url.startswith(("http://", "https://")):
            raise ValueError("api_url must be an HTTP(S) URL")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")


class OpenAICompatibleGenerator:
    """Request JSON proposals from a hosted OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: ApiGeneratorConfig,
        transport: Optional[Transport] = None,
    ) -> None:
        self.config = config
        self.transport = transport or _post_json

    def generate(
        self,
        task: TaskSpec,
        parent: Candidate,
        evidence: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
        count: int,
    ) -> List[CandidateProposal]:
        api_key = os.environ.get(self.config.api_key_environment_variable)
        if not api_key:
            raise RuntimeError(
                "API key environment variable %s is not set"
                % self.config.api_key_environment_variable
            )
        prompt = build_optimization_prompt(task, parent, evidence, history, count)
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.temperature,
        }
        response = self.transport(
            self.config.api_url,
            {
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
            payload,
            self.config.timeout_seconds,
        )
        content = self._response_content(response)
        data = json.loads(_strip_json_fence(content))
        raw_candidates = data.get("candidates") if isinstance(data, dict) else data
        if not isinstance(raw_candidates, list):
            raise ValueError("API response must contain a candidates list")
        proposals = []
        for raw in raw_candidates[:count]:
            if not isinstance(raw, dict):
                raise ValueError("each API candidate must be a JSON object")
            value = dict(raw)
            if "parameter_updates" not in value and "parameters" in value:
                value["parameter_updates"] = value.pop("parameters")
            proposals.append(CandidateProposal.from_dict(value))
        if not proposals:
            raise ValueError("API response did not contain any candidate proposals")
        return proposals

    @staticmethod
    def _response_content(response: Mapping[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("API response is missing choices[0].message.content") from error
        if not isinstance(content, str) or not content.strip():
            raise ValueError("API response content must be a non-empty string")
        return content


def _strip_json_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, re.DOTALL)
    return match.group(1) if match else text.strip()


def _post_json(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            "hosted API returned HTTP %d: %s" % (error.code, body[:1000])
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError("hosted API request failed: %s" % error.reason) from error
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("hosted API response must be a JSON object")
    return parsed

