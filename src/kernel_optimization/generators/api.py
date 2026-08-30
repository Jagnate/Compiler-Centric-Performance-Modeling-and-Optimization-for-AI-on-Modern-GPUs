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
    COMPRESSION_VERSION,
    METADATA_COMPRESSION_POLICIES,
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
    metadata_compression_policy: str = COMPRESSION_VERSION

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
        if self.metadata_compression_policy not in METADATA_COMPRESSION_POLICIES:
            raise ValueError(
                "metadata_compression_policy must be one of: %s"
                % ", ".join(METADATA_COMPRESSION_POLICIES)
            )
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
        prompt = build_optimization_prompt(
            task,
            parent,
            evidence,
            history,
            count,
            metadata_compression_policy=self.config.metadata_compression_policy,
        )
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
            metadata_compression_policy=self.config.metadata_compression_policy,
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
        proposals, parsing = _parse_candidate_response(data, count=count)
        self._record_candidate_parsing(parsing)
        if proposals:
            return proposals

        first_metadata = dict(self.last_call_metadata)
        first_exchange = dict(self.last_exchange)
        recovery_prompt = _candidate_schema_recovery_prompt(prompt)
        try:
            recovery_data = self._request_json_prompt(
                recovery_prompt,
                system_prompt=SYSTEM_PROMPT,
                kind=kind + "-schema-recovery",
                max_output_tokens=self.config.max_output_tokens,
            )
            recovered, recovery_parsing = _parse_candidate_response(
                recovery_data, count=1
            )
            second_metadata = dict(self.last_call_metadata)
            second_exchange = dict(self.last_exchange)
        except Exception:
            self._merge_schema_attempts(
                kind,
                (first_metadata, dict(self.last_call_metadata)),
                (first_exchange, dict(self.last_exchange)),
                (parsing,),
            )
            raise

        self._merge_schema_attempts(
            kind,
            (first_metadata, second_metadata),
            (first_exchange, second_exchange),
            (parsing, recovery_parsing),
        )
        if recovered:
            return recovered
        raise ValueError(
            "API candidate response remained unusable after one schema-recovery "
            "request: no candidate supplied a non-empty complete source file"
        )

    def _record_candidate_parsing(self, parsing: Mapping[str, Any]) -> None:
        metadata = dict(self.last_call_metadata)
        metadata["candidate_parsing"] = dict(parsing)
        self.last_call_metadata = metadata
        exchange = dict(self.last_exchange)
        exchange["candidate_parsing"] = dict(parsing)
        self.last_exchange = exchange

    def _merge_schema_attempts(
        self,
        kind: str,
        metadata_items: Sequence[Mapping[str, Any]],
        exchange_items: Sequence[Mapping[str, Any]],
        parsing_items: Sequence[Mapping[str, Any]],
    ) -> None:
        merged = dict(metadata_items[-1]) if metadata_items else {}
        merged["kind"] = kind
        merged["attempts"] = sum(
            int(item.get("attempts", 0) or 0) for item in metadata_items
        )
        merged["elapsed_seconds"] = sum(
            float(item.get("elapsed_seconds", 0.0) or 0.0)
            for item in metadata_items
        )
        merged["usage"] = _sum_usage(
            dict(item.get("usage") or {}) for item in metadata_items
        )
        merged["response_validation_attempts"] = len(metadata_items)
        merged["response_ids"] = [
            item.get("response_id")
            for item in metadata_items
            if item.get("response_id") is not None
        ]
        merged["candidate_parsing_attempts"] = [
            dict(item) for item in parsing_items
        ]
        if parsing_items:
            merged["candidate_parsing"] = dict(parsing_items[-1])
        self.last_call_metadata = merged
        self.last_exchange = {
            "kind": kind,
            "response_validation_attempts": [
                dict(item) for item in exchange_items
            ],
            "candidate_parsing_attempts": [
                dict(item) for item in parsing_items
            ],
        }

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


_SOURCE_FIELDS = (
    "source_code",
    "source",
    "code",
    "kernel_source",
    "complete_source",
    "implementation",
)
_HYPOTHESIS_FIELDS = ("hypothesis", "rationale", "description", "optimization")
_EXPECTED_EFFECT_FIELDS = ("expected_effect", "expected_impact", "effect")
_METADATA_FIELDS = (
    "strategy",
    "strategy_slot",
    "strategy_id",
    "discovered_strategy",
    "related_existing_strategies",
    "changed_regions",
    "evidence_ids",
    "applied_recommendations",
    "structural_required",
)


def _parse_candidate_response(
    data: Any, *, count: int
) -> tuple[List[CandidateProposal], Dict[str, Any]]:
    try:
        raw_candidates, container = _candidate_payloads(data)
    except ValueError as error:
        return [], {
            "container": "invalid",
            "requested_candidates": count,
            "returned_candidates": 0,
            "accepted_candidates": 0,
            "rejected_candidates": 1,
            "rejections": [
                {
                    "index": None,
                    "value_type": type(data).__name__,
                    "keys": sorted(str(key) for key in data)
                    if isinstance(data, dict)
                    else [],
                    "error": str(error),
                }
            ],
            "normalizations": [],
        }
    proposals: List[CandidateProposal] = []
    rejections: List[Dict[str, Any]] = []
    normalizations: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates):
        if len(proposals) >= count:
            break
        try:
            proposal, normalization = _parse_candidate_payload(raw)
        except (TypeError, ValueError) as error:
            rejections.append(
                {
                    "index": index,
                    "value_type": type(raw).__name__,
                    "keys": sorted(str(key) for key in raw) if isinstance(raw, dict) else [],
                    "error": str(error),
                }
            )
            continue
        proposals.append(proposal)
        if normalization:
            normalization["index"] = index
            normalizations.append(normalization)
    return proposals, {
        "container": container,
        "requested_candidates": count,
        "returned_candidates": len(raw_candidates),
        "accepted_candidates": len(proposals),
        "rejected_candidates": len(rejections),
        "rejections": rejections,
        "normalizations": normalizations,
    }


def _candidate_payloads(data: Any) -> tuple[List[Any], str]:
    if isinstance(data, list):
        return list(data), "top-level-list"
    if not isinstance(data, dict):
        raise ValueError("API response must be a JSON object or candidate list")
    for key in ("candidates", "proposals", "results"):
        value = data.get(key)
        if isinstance(value, list):
            return list(value), key
        if isinstance(value, dict):
            return [value], key + "-object"
    candidate = data.get("candidate")
    if isinstance(candidate, dict):
        return [candidate], "candidate-object"
    if any(key in data for key in _SOURCE_FIELDS + _HYPOTHESIS_FIELDS):
        return [data], "top-level-candidate"
    raise ValueError(
        "API response must contain candidates, proposals, results, or candidate"
    )


def _parse_candidate_payload(
    raw: Any,
) -> tuple[CandidateProposal, Dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("candidate must be a JSON object")
    contexts: List[tuple[str, Mapping[str, Any]]] = [("candidate", raw)]
    for wrapper in ("candidate", "proposal", "implementation"):
        nested = raw.get(wrapper)
        if isinstance(nested, dict):
            contexts.append((wrapper, nested))

    source_code = None
    source_field = None
    for context_name, context in contexts:
        for field in _SOURCE_FIELDS:
            candidate_source = _source_text(context.get(field))
            if candidate_source is not None:
                source_code = _strip_source_fence(candidate_source)
                source_field = "%s.%s" % (context_name, field)
                break
        if source_code is not None:
            break
    if source_code is None or not source_code.strip():
        raise ValueError(
            "missing non-empty source_code; accepted source fields are %s"
            % ", ".join(_SOURCE_FIELDS)
        )

    hypothesis = _first_text(contexts, _HYPOTHESIS_FIELDS)
    if hypothesis is None:
        hypothesis = "Hosted API source proposal."

    expected_effect: Dict[str, Any] = {}
    for _context_name, context in contexts:
        for field in _EXPECTED_EFFECT_FIELDS:
            value = context.get(field)
            if isinstance(value, dict):
                expected_effect = dict(value)
                break
            if isinstance(value, str) and value.strip():
                expected_effect = {"reason": value.strip()}
                break
        if expected_effect:
            break

    metadata: Dict[str, Any] = {}
    for _context_name, context in reversed(contexts):
        value = context.get("metadata")
        if isinstance(value, dict):
            metadata.update(value)
    for _context_name, context in contexts:
        for field in _METADATA_FIELDS:
            if field in context and field not in metadata:
                metadata[field] = context[field]
    normalization: Dict[str, Any] = {}
    if source_field != "candidate.source_code":
        normalization["source_field"] = source_field
        metadata["response_source_field"] = source_field
    if not any(
        isinstance(context.get("hypothesis"), str)
        and context.get("hypothesis").strip()
        for _name, context in contexts
    ):
        normalization["hypothesis_fallback"] = hypothesis

    return (
        CandidateProposal(
            hypothesis=hypothesis,
            source_code=source_code,
            expected_effect=expected_effect,
            metadata=metadata,
        ),
        normalization,
    )


def _source_text(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, dict):
        for field in ("content", "source_code", "source", "code", "text"):
            nested = value.get(field)
            if isinstance(nested, str) and nested.strip():
                return nested
    return None


def _first_text(
    contexts: Sequence[tuple[str, Mapping[str, Any]]], fields: Sequence[str]
) -> Optional[str]:
    for _context_name, context in contexts:
        for field in fields:
            value = context.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _candidate_schema_recovery_prompt(prompt: str) -> str:
    try:
        request = json.loads(prompt)
    except (TypeError, json.JSONDecodeError):
        return prompt + (
            "\nReturn one candidate containing a non-empty source_code with the "
            "complete Python source. Do not return a patch, placeholder, or "
            "metadata-only candidate."
        )
    request["candidate_count"] = 1
    assignments = request.get("strategy_assignments")
    if isinstance(assignments, list):
        request["strategy_assignments"] = assignments[:1]
    rules = list(request.get("rules") or [])
    rules.insert(
        0,
        "Return exactly one candidate with a non-empty source_code containing the complete Python file.",
    )
    rules.insert(
        1,
        "Never omit source_code or replace it with a patch, placeholder, summary, or unchanged marker.",
    )
    request["rules"] = rules
    request["response_recovery"] = {
        "reason": "The previous response contained no usable complete source file.",
        "required_action": (
            "Return one fully materialized candidate in the documented schema."
        ),
    }
    return json.dumps(request, indent=2, sort_keys=True)


def _sum_usage(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, Any] = {}
    for item in items:
        for key, value in item.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
            elif key not in totals:
                totals[key] = value
    return totals


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
