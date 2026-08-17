"""Hosted OpenAI-compatible source-code candidate generator."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..prompt_compression import (
    PromptBudgetError,
    enforce_prompt_budget,
    prompt_size_metadata,
)
from ..prompts import (
    STRATEGY_PLANNER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_optimization_prompt,
    build_repair_prompt,
    build_strategy_planning_prompt,
)
from ..schema import Candidate, CandidateProposal, TaskSpec


Transport = Callable[
    [str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]
]
RequestObserver = Callable[[Mapping[str, Any]], None]


class HostedApiError(RuntimeError):
    """A classified provider failure with enough information for retry policy."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None,
        retryable: bool = False,
        retry_after_seconds: Optional[float] = None,
        response_body: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.response_body = response_body

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": type(self).__name__,
            "message": str(self),
            "status_code": self.status_code,
            "error_code": self.error_code,
            "retryable": self.retryable,
            "retry_after_seconds": self.retry_after_seconds,
            "response_body": self.response_body,
        }


@dataclass(frozen=True)
class ApiGeneratorConfig:
    """Connection, output, and retry settings for a chat-completions API."""

    api_url: str
    model: str
    api_key_environment_variable: str = "KERNEL_OPT_API_KEY"
    timeout_seconds: float = 120.0
    temperature: Optional[float] = None
    max_output_tokens: Optional[int] = 12000
    planner_max_output_tokens: int = 2000
    max_tokens_field: str = "max_completion_tokens"
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    use_json_object: bool = True
    max_input_tokens: int = 60000

    def __post_init__(self) -> None:
        if not self.api_url.startswith(("http://", "https://")):
            raise ValueError("api_url must be an HTTP(S) URL")
        if not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive when provided")
        if self.planner_max_output_tokens <= 0:
            raise ValueError("planner_max_output_tokens must be positive")
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive")
        if self.max_tokens_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "max_tokens_field must be max_tokens or max_completion_tokens"
            )
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")


class OpenAICompatibleGenerator:
    """Request JSON proposals from a hosted OpenAI-compatible endpoint."""

    def __init__(
        self,
        config: ApiGeneratorConfig,
        transport: Optional[Transport] = None,
        sleeper: Callable[[float], None] = time.sleep,
        request_observer: Optional[RequestObserver] = None,
    ) -> None:
        self.config = config
        self.transport = transport or _post_json
        self.sleeper = sleeper
        self.request_observer = request_observer
        self.last_call_metadata: Dict[str, Any] = {}
        self.last_exchange: Dict[str, Any] = {}
        self.total_api_requests = 0

    def preflight(self) -> Dict[str, Any]:
        """Perform a tiny request before any expensive compiler or GPU work."""

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return a JSON object and no additional text.",
                },
                {"role": "user", "content": '{"health_check": true}'},
            ],
        }
        if self.config.use_json_object:
            payload["response_format"] = {"type": "json_object"}
        payload[self.config.max_tokens_field] = min(
            self.config.max_output_tokens or 128, 128
        )
        size = prompt_size_metadata(
            payload["messages"][0]["content"],
            payload["messages"][1]["content"],
            payload[self.config.max_tokens_field],
        )
        self._observe_request("preflight", size)
        response = self._request(payload, kind="preflight", prompt_size=size)
        return dict(self.last_call_metadata)

    def generate(
        self,
        task: TaskSpec,
        parent: Candidate,
        evidence: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
        count: int,
    ) -> List[CandidateProposal]:
        prompt = build_optimization_prompt(task, parent, evidence, history, count)
        return self._generate_from_prompt(prompt, count=count, kind="generate")

    def plan_strategies(
        self,
        task: TaskSpec,
        parent: Candidate,
        planning_context: Dict[str, Any],
        strategies: Sequence[Dict[str, Any]],
        count: int,
        round_number: int,
    ) -> Dict[str, Any]:
        """Ask the hosted model for a compact pre-generation slot allocation."""

        prompt = build_strategy_planning_prompt(
            task,
            parent,
            planning_context,
            strategies,
            count,
            round_number,
        )
        data = self._request_json_prompt(
            prompt,
            system_prompt=STRATEGY_PLANNER_SYSTEM_PROMPT,
            kind="plan",
            max_output_tokens=min(
                self.config.planner_max_output_tokens,
                self.config.max_output_tokens
                if self.config.max_output_tokens is not None
                else self.config.planner_max_output_tokens,
            ),
        )
        if not isinstance(data, dict):
            raise ValueError("strategy planner response must be a JSON object")
        if not isinstance(data.get("allocation"), list):
            raise ValueError("strategy planner response must contain an allocation list")
        return dict(data)

    def repair(
        self,
        task: TaskSpec,
        failed: Candidate,
        failure: Dict[str, Any],
        evidence: Dict[str, Any],
        history: Sequence[Dict[str, Any]],
    ) -> CandidateProposal:
        prompt = build_repair_prompt(
            task,
            failed,
            failure,
            evidence,
            history,
        )
        proposals = self._generate_from_prompt(prompt, count=1, kind="repair")
        return proposals[0]

    def _generate_from_prompt(
        self, prompt: str, *, count: int, kind: str
    ) -> List[CandidateProposal]:
        data = self._request_json_prompt(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            kind=kind,
            max_output_tokens=self.config.max_output_tokens,
        )
        raw_candidates = data.get("candidates") if isinstance(data, dict) else data
        if not isinstance(raw_candidates, list):
            raise ValueError("API response must contain a candidates list")
        proposals = []
        for raw in raw_candidates[:count]:
            if not isinstance(raw, dict):
                raise ValueError("each API candidate must be a JSON object")
            value = dict(raw)
            source_code = value.get("source_code")
            if isinstance(source_code, str):
                value["source_code"] = _strip_source_fence(source_code)
            proposals.append(CandidateProposal.from_dict(value))
        if not proposals:
            raise ValueError("API response did not contain any candidate proposals")
        return proposals

    def _request_json_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str,
        kind: str,
        max_output_tokens: Optional[int],
    ) -> Any:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.use_json_object:
            payload["response_format"] = {"type": "json_object"}
        if max_output_tokens is not None:
            payload[self.config.max_tokens_field] = max_output_tokens
        size = prompt_size_metadata(
            system_prompt,
            prompt,
            max_output_tokens,
        )
        self._observe_request(kind, size)
        try:
            enforce_prompt_budget(size, self.config.max_input_tokens)
        except PromptBudgetError as error:
            error_metadata = {
                "type": type(error).__name__,
                "message": str(error),
                "error_code": "local_prompt_budget_exceeded",
                "retryable": False,
            }
            self.last_call_metadata = {
                "kind": kind,
                "requested_model": self.config.model,
                "attempts": 0,
                "prompt_size": dict(size),
                "error": error_metadata,
            }
            self.last_exchange = {
                "request": dict(payload),
                "kind": kind,
                "prompt_size": dict(size),
                "error": error_metadata,
            }
            raise
        response = self._request(payload, kind=kind, prompt_size=size)
        content = self._response_content(response)
        return json.loads(_strip_json_fence(content))

    def _request(
        self,
        payload: Mapping[str, Any],
        *,
        kind: str,
        prompt_size: Optional[Mapping[str, Any]] = None,
    ) -> Mapping[str, Any]:
        api_key = os.environ.get(self.config.api_key_environment_variable)
        if not api_key:
            raise RuntimeError(
                "API key environment variable %s is not set"
                % self.config.api_key_environment_variable
            )
        headers = {
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        }
        started_at = time.perf_counter()
        attempts = 0
        self.last_exchange = {
            "request": dict(payload),
            "kind": kind,
            "prompt_size": dict(prompt_size or {}),
        }
        while True:
            attempts += 1
            self.total_api_requests += 1
            try:
                response = self.transport(
                    self.config.api_url,
                    headers,
                    payload,
                    self.config.timeout_seconds,
                )
                break
            except HostedApiError as error:
                self.last_call_metadata = {
                    "kind": kind,
                    "requested_model": self.config.model,
                    "attempts": attempts,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "prompt_size": dict(prompt_size or {}),
                    "error": error.to_dict(),
                }
                self.last_exchange["error"] = error.to_dict()
                if not error.retryable or attempts > self.config.max_retries:
                    raise
                delay = error.retry_after_seconds
                if delay is None:
                    delay = self.config.retry_backoff_seconds * (2 ** (attempts - 1))
                self.sleeper(delay)

        choices = response.get("choices") or []
        finish_reason = (
            choices[0].get("finish_reason")
            if choices and isinstance(choices[0], dict)
            else None
        )
        self.last_call_metadata = {
            "kind": kind,
            "response_id": response.get("id"),
            "requested_model": self.config.model,
            "response_model": response.get("model"),
            "finish_reason": finish_reason,
            "usage": dict(response.get("usage") or {}),
            "attempts": attempts,
            "elapsed_seconds": time.perf_counter() - started_at,
            "prompt_size": dict(prompt_size or {}),
        }
        self.last_exchange["response"] = dict(response)
        return response

    def _observe_request(self, kind: str, size: Mapping[str, Any]) -> None:
        if self.request_observer is None:
            return
        value = dict(size)
        value["kind"] = kind
        value["max_input_tokens"] = self.config.max_input_tokens
        self.request_observer(value)

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


def _strip_source_fence(text: str) -> str:
    match = re.fullmatch(r"\s*```(?:python|py)?\s*(.*?)\s*```\s*", text, re.DOTALL)
    if match:
        return match.group(1).rstrip() + "\n"
    return text


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
        error_code = _error_code(body)
        retryable = error.code >= 500 or (
            error.code == 429
            and error_code not in {"insufficient_quota", "credit_balance_exhausted"}
            and not _request_too_large(body)
        )
        retry_after = _retry_after(error.headers.get("Retry-After"))
        raise HostedApiError(
            "hosted API returned HTTP %d: %s" % (error.code, body[:1000]),
            status_code=error.code,
            error_code=error_code,
            retryable=retryable,
            retry_after_seconds=retry_after,
            response_body=body[:4000],
        ) from error
    except urllib.error.URLError as error:
        raise HostedApiError(
            "hosted API request failed: %s" % error.reason,
            retryable=True,
        ) from error
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("hosted API response must be a JSON object")
    return parsed


def _error_code(body: str) -> Optional[str]:
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if not isinstance(error, dict):
        return None
    code = error.get("code") or error.get("type")
    return str(code) if code is not None else None


def _request_too_large(body: str) -> bool:
    lowered = body.lower()
    return "request too large" in lowered or (
        "input or output tokens" in lowered and "must be reduced" in lowered
    )


def _retry_after(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
