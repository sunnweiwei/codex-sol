"""
Proposed production-grade implementation for codex.model.
Provides ModelStreamEvent, ScriptedResponsesModel, collect_stream_response, 
iter_model_stream_events, load_env_file, and custom runtime clients/exceptions.
"""

from __future__ import annotations

import json
import os
import random
import socket
import time
import uuid
import threading
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Sequence, Iterable

from codex.types import ModelResponse, PromptRequest


class ModelRuntimeError(Exception):
    """Base exception for custom model runtime failures."""
    pass


class ModelRateLimitError(ModelRuntimeError):
    """Raised when the model API returns a 429 Rate Limit Exceeded error."""
    pass


class ModelContextLengthError(ModelRuntimeError):
    """Raised when the prompt token usage exceeds context window limits."""
    pass


class ModelStreamEvent:
    """
    Represents an individual event frame emitted by the model during a streaming turn execution.
    """
    def __init__(self, type: str, payload: dict[str, Any]) -> None:
        self._type = type
        self._payload = payload

    @property
    def type(self) -> str:
        """The type identifier of this stream event."""
        return self._type

    @property
    def payload(self) -> dict[str, Any]:
        """The raw data dictionary associated with this stream event."""
        return self._payload

    @property
    def text(self) -> str | None:
        """Extracts text content or text delta from this stream event if applicable."""
        # Check standard Responses API payload keys
        if "delta" in self._payload and isinstance(self._payload["delta"], str):
            return self._payload["delta"]
        if "text" in self._payload and isinstance(self._payload["text"], str):
            return self._payload["text"]
        # Nested output item data structure
        item = self._payload.get("item", {})
        if isinstance(item, dict):
            content_list = item.get("content", [])
            if isinstance(content_list, list):
                texts = []
                for chunk in content_list:
                    if isinstance(chunk, dict) and "text" in chunk:
                        texts.append(chunk["text"])
                if texts:
                    return "".join(texts)
        return None

    @property
    def tool_call_delta(self) -> dict[str, Any] | None:
        """Extracts tool call details and argument delta if applicable."""
        if self._type == "response.custom_tool_call_input.delta":
            return {
                "call_id": self._payload.get("call_id"),
                "item_id": self._payload.get("item_id"),
                "delta": self._payload.get("delta", "")
            }
        return None

    @property
    def usage(self) -> dict[str, Any] | None:
        """Extracts token usage metrics from the completed turn block."""
        # Top-level usage dictionary
        if "usage" in self._payload and isinstance(self._payload["usage"], dict):
            return self._payload["usage"]
        # Nested under completed response structure
        resp = self._payload.get("response", {})
        if isinstance(resp, dict) and "usage" in resp and isinstance(resp["usage"], dict):
            return resp["usage"]
        return None

    @property
    def is_complete(self) -> bool:
        """Returns True if this event signals final completion of the stream."""
        return self._type in ("turn_complete", "response.completed", "response.failed", "error")

    @property
    def is_error(self) -> bool:
        """Returns True if this event represents a failure signal."""
        return self._type in ("error", "response.failed", "stream_error")

    @property
    def error_message(self) -> str | None:
        """Retrieves the error detail string if the event is a failure signal."""
        if not self.is_error:
            return None
        # Root-level error key
        err = self._payload.get("error")
        if isinstance(err, dict):
            return err.get("message")
        # Nested response error block
        resp = self._payload.get("response", {})
        if isinstance(resp, dict):
            resp_err = resp.get("error")
            if isinstance(resp_err, dict):
                return resp_err.get("message")
        # Direct string fallbacks
        if isinstance(err, str):
            return err
        return self._payload.get("delta") or self._payload.get("text") or "Stream execution failed"


# Thread-local registry to verify duplicate overwrite and active mocks in E2E tests
_ACTIVE_MOCKS: dict[str, ScriptedResponsesModel] = {}
_MOCK_REGISTRY_LOCK = threading.Lock()


class ScriptedResponsesModel:
    """
    In-memory mocked client for E2E tests and local playback validation.
    Maintains chronological turn matches, registers calls, and yields realistic deltas.
    """
    def __init__(self, responses: Sequence[dict[str, Any]]) -> None:
        # Register blanks validation: empty inputs/dicts raise value/type errors
        if not responses:
            raise ValueError("Mock responses sequence cannot be empty")
        
        self._responses = []
        for idx, r in enumerate(responses):
            if not isinstance(r, dict):
                raise TypeError(f"Response at index {idx} must be a dictionary, got {type(r)}")
            if not r:
                raise ValueError(f"Response dictionary at index {idx} cannot be empty")
            self._responses.append(r)

        self._requests = []
        self._current_index = 0
        self._lock = threading.Lock()

    @property
    def requests(self) -> list[dict[str, Any]]:
        """A collection of all requests registered by this player."""
        with self._lock:
            # Emulate standard read-only tracking property
            return list(self._requests)

    def stream(self, request: PromptRequest | dict[str, Any]) -> Iterable[ModelStreamEvent]:
        """
        Intercepts model streams, processes incoming calls, and yields realistic deltas.
        Matches sequentially, enforces thread concurrency locks, and detects bounds exhaustions.
        """
        with self._lock:
            # 1. Record incoming request details in self.requests property
            if hasattr(request, "to_compact_payload"):
                self._requests.append(request.to_compact_payload())
            else:
                self._requests.append(request)

            # 2. Out of bounds check: throw error when pre-stored events run out
            if self._current_index >= len(self._responses):
                raise IndexError("Mock sequential responses matching failed: Playback sequence exhausted")

            response_turn = self._responses[self._current_index]
            self._current_index += 1

        # Yielding standard event deltas under offline setups
        # If mock turn is directly a list of event frames:
        if isinstance(response_turn.get("events"), list):
            for raw_ev in response_turn["events"]:
                yield ModelStreamEvent(
                    type=raw_ev.get("event") or raw_ev.get("type") or "message",
                    payload=raw_ev.get("payload") or raw_ev
                )
        else:
            # Emulate realistic Responses API stream from basic dictionary input (e.g. {"text": "A"})
            yield ModelStreamEvent(
                type="response.created",
                payload={"response": {"id": "mock_resp_session"}}
            )
            
            text_val = response_turn.get("text", "")
            if text_val:
                # Chunk split for delta simulation
                chunk_size = max(1, len(text_val) // 3)
                for i in range(0, len(text_val), chunk_size):
                    chunk = text_val[i:i+chunk_size]
                    yield ModelStreamEvent(
                        type="response.output_text.delta",
                        payload={"type": "response.output_text.delta", "delta": chunk}
                    )

            # Check if custom mock tool calls exist
            tool_calls = response_turn.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    cid = tc.get("call_id", str(uuid.uuid4()))
                    args_text = json.dumps(tc.get("arguments", {}))
                    # yield custom tool call delta chunks
                    chunk_size = max(1, len(args_text) // 2)
                    for i in range(0, len(args_text), chunk_size):
                        chunk = args_text[i:i+chunk_size]
                        yield ModelStreamEvent(
                            type="response.custom_tool_call_input.delta",
                            payload={
                                "type": "response.custom_tool_call_input.delta",
                                "call_id": cid,
                                "item_id": tc.get("item_id"),
                                "delta": chunk
                            }
                        )

            # Signal completion
            yield ModelStreamEvent(
                type="response.completed",
                payload={
                    "response": {
                        "id": "mock_resp_session",
                        "usage": {"input_tokens": 120, "output_tokens": 80}
                    }
                }
            )

    def __call__(self, request: PromptRequest | dict[str, Any]) -> Iterable[ModelStreamEvent]:
        """Provides direct callable support for standard model run loops."""
        return self.stream(request)

    def generate(self, request: PromptRequest | dict[str, Any]) -> ModelResponse:
        """Blocks and compiles output to a unified ModelResponse return object."""
        return collect_stream_response(self.stream(request))


def register_mock(id_key: str, mock_model: ScriptedResponsesModel) -> None:
    """
    Registers a mock responses model instance for E2E testing setups.
    Overwrites the mock registry on duplicate key detection.
    """
    if not id_key or not isinstance(id_key, str):
        raise ValueError("Invalid id key: registry lookup key must be a non-empty string")
    
    with _MOCK_REGISTRY_LOCK:
        _ACTIVE_MOCKS[id_key] = mock_model


def clear_mock_registry() -> None:
    """Clears active offline player mocks."""
    with _MOCK_REGISTRY_LOCK:
        _ACTIVE_MOCKS.clear()


class ResponsesApiClient:
    """
    Production-grade client for the live OpenAI Responses API.
    Provides exponential backoff retries, header tracking, and SSE decoding.
    """
    def __init__(self, api_key: str | None = None, base_url: str | None = None, max_retries: int = 5, base_delay_ms: int = 200) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        env_url = os.environ.get("PY_CODEX_RESPONSES_API_URL")
        if env_url:
            self.base_url = env_url
        else:
            base = base_url or os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1"
            self.base_url = base.rstrip('/') + "/responses"
            
        self.max_retries = max_retries
        self.base_delay_ms = base_delay_ms

    def stream(self, request: PromptRequest | dict[str, Any]) -> Iterable[ModelStreamEvent]:
        """
        Executes a dynamic streaming request using standard library HTTP transports.
        Applies rate limit retries and maps exceptions based on HTTP response blocks.
        """
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ModelRuntimeError("Authentication failure: OPENAI_API_KEY environment variable is not configured")

        if hasattr(request, "to_compact_payload"):
            payload = request.to_compact_payload()
            model_name = request.model
        else:
            payload = request
            model_name = request.get("model", "gpt-5.5")

        body_bytes = json.dumps(payload).encode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "openai-model": model_name,
            "x-request-id": str(uuid.uuid4())
        }

        # Custom metadata/header tracking injection
        if hasattr(request, "client_metadata") and request.client_metadata:
            for k, v in request.client_metadata.items():
                headers[k] = v

        attempt = 0
        while True:
            try:
                req = urllib.request.Request(
                    url=self.base_url,
                    data=body_bytes,
                    headers=headers,
                    method="POST"
                )
                
                # Standard library stream transport connection
                response = urllib.request.urlopen(req, timeout=30.0)
                yield from iter_model_stream_events(response)
                return # Completed successfully

            except urllib.error.HTTPError as e:
                status = e.code
                is_retryable = (status == 429) or (500 <= status < 600)
                
                if is_retryable and attempt < self.max_retries:
                    # Exponential delay with jitter
                    attempt += 1
                    exp = 2 ** (attempt - 1)
                    raw_delay = (self.base_delay_ms * exp) / 1000.0
                    jitter = random.uniform(0.9, 1.1)
                    sleep_sec = raw_delay * jitter
                    time.sleep(sleep_sec)
                    continue
                else:
                    error_msg = f"HTTP Error {status}: {e.reason}"
                    try:
                        body = e.read().decode("utf-8")
                        payload = json.loads(body)
                        if "error" in payload:
                            error_msg = payload["error"].get("message", error_msg)
                    except Exception:
                        pass
                        
                    if status == 429:
                        raise ModelRateLimitError(f"Rate limit exceeded on OpenAI API connection: {error_msg}")
                    elif status == 400 and ("context_length_exceeded" in error_msg or "context window" in error_msg.lower()):
                        raise ModelContextLengthError(f"Prompt token count exceeds the active context window boundaries: {error_msg}")
                    else:
                        raise ModelRuntimeError(f"API Execution turn failure: {error_msg}")
                        
            except (socket.timeout, socket.gaierror, urllib.error.URLError, ConnectionError, TimeoutError) as e:
                # Transport failures are retryable
                if attempt < self.max_retries:
                    attempt += 1
                    exp = 2 ** (attempt - 1)
                    raw_delay = (self.base_delay_ms * exp) / 1000.0
                    jitter = random.uniform(0.9, 1.1)
                    sleep_sec = raw_delay * jitter
                    time.sleep(sleep_sec)
                    continue
                else:
                    raise ModelRuntimeError(f"Network transport timeout or error: {str(e)}")


def collect_stream_response(events: Iterable[Any]) -> ModelResponse:
    """
    Takes stream event structures (or ModelStreamEvent instances) and aggregates text deltas, 
    tool input deltas, usage metadata, and raw parameters into a unified ModelResponse return container.
    """
    aggregated_text = []
    tool_calls = {}
    usage = {}
    response_id = "collected_stream_response"

    for raw_ev in events:
        # Standardize dynamic types
        if isinstance(raw_ev, ModelStreamEvent):
            ev = raw_ev
        elif isinstance(raw_ev, dict):
            ev = ModelStreamEvent(type=raw_ev.get("type", "message"), payload=raw_ev.get("payload", raw_ev))
        else:
            continue

        # 1. Text Deltas mapping
        if ev.type in ("response.output_text.delta", "text_delta", "agent_message"):
            text = ev.text
            if text:
                aggregated_text.append(text)

        # 2. Tool call argument deltas
        elif ev.type in ("response.custom_tool_call_input.delta", "tool_call_delta"):
            tool_delta = ev.tool_call_delta
            if tool_delta:
                cid = tool_delta.get("call_id") or "default_call"
                iid = tool_delta.get("item_id")
                delta = tool_delta.get("delta", "")
                
                if cid not in tool_calls:
                    tool_calls[cid] = {
                        "type": "tool_search_call",
                        "call_id": cid,
                        "item_id": iid,
                        "arguments": ""
                    }
                tool_calls[cid]["arguments"] += delta

        # 3. Usage metadata
        elif ev.type in ("response.completed", "turn_complete"):
            resp_usage = ev.usage
            if resp_usage:
                usage.update(resp_usage)
            resp_id = ev.payload.get("response", {}).get("id")
            if resp_id:
                response_id = resp_id

        elif ev.type == "response.created":
            resp_id = ev.payload.get("response", {}).get("id")
            if resp_id:
                response_id = resp_id

    # Construct the standard output list structure
    output_items = []
    if aggregated_text:
        output_items.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "".join(aggregated_text)}]
        })

    for cid, tc in tool_calls.items():
        arg_str = tc["arguments"]
        try:
            parsed_args = json.loads(arg_str)
        except Exception:
            parsed_args = arg_str
        output_items.append({
            "type": "tool_search_call",
            "call_id": cid,
            "arguments": parsed_args
        })

    return ModelResponse(
        id=response_id,
        output=output_items,
        raw={"usage": usage}
    )


def iter_model_stream_events(events: Iterable[Any]) -> Iterable[ModelStreamEvent]:
    """
    Reads an iterable sequence of raw string chunks or bytes streams, splits them 
    according to Server-Sent Events standards (preserving trailing block residues and multiline splits), 
    filters keep-alive ticks/comments, decodes JSON structures, handles stream error blocks, 
    and yields fully instantiated ModelStreamEvent instances.
    """
    buffer = ""
    current_event_type = ""
    current_data = []

    for chunk in events:
        if isinstance(chunk, bytes):
            text_chunk = chunk.decode("utf-8")
        elif isinstance(chunk, str):
            text_chunk = chunk
        else:
            text_chunk = str(chunk)

        buffer += text_chunk
        
        # Standardize all CR/CRLF -> LF and split on newlines
        normalized = buffer.replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized.split("\n")
        
        # Preserve the uncompleted trailing line residue
        buffer = lines.pop()

        for line in lines:
            line_stripped = line.strip()
            
            # 1. Dispatch on double newline boundary signal (empty line)
            if not line_stripped:
                if current_data:
                    data_str = "\n".join(current_data)
                    current_data = []
                    
                    # Gracefully catch invalid JSON and skip, but support fallback for plain text data blocks
                    try:
                        is_json_candidate = (
                            data_str.startswith(("{", "[", "\"", "'")) or
                            data_str.isdigit() or
                            data_str in ("true", "false", "null")
                        )
                        if is_json_candidate:
                            payload = json.loads(data_str)
                        else:
                            payload = {"text": data_str}
                    except json.JSONDecodeError:
                        current_event_type = ""
                        continue

                    event_type = current_event_type or payload.get("type") or "message"
                    current_event_type = ""

                    # Error handling: failed API SSE signals mapped to custom model exceptions
                    is_error = (
                        event_type in ("error", "stream_error", "response.failed") or 
                        "error" in payload
                    )
                    if is_error:
                        err_msg = ""
                        err_code = None
                        
                        if "error" in payload:
                            err_msg = payload["error"].get("message", "")
                            err_code = payload["error"].get("code")
                        elif "response" in payload and "error" in payload["response"]:
                            err_msg = payload["response"]["error"].get("message", "")
                            err_code = payload["response"]["error"].get("code")
                        else:
                            err_msg = payload.get("delta") or payload.get("text") or str(payload)

                        # Match exact error conditions
                        if err_code == "rate_limit_exceeded" or "rate limit" in err_msg.lower():
                            raise ModelRateLimitError(f"Rate Limit Exceeded: {err_msg}")
                        elif err_code == "context_length_exceeded" or "context length" in err_msg.lower() or "context window" in err_msg.lower():
                            raise ModelContextLengthError(f"Context Length Exceeded: {err_msg}")
                        else:
                            raise ModelRuntimeError(f"Model SSE error frame returned: {err_msg}")

                    yield ModelStreamEvent(type=event_type, payload=payload)
                continue

            # 2. Skip keep-alive/comment ticks starting with colon
            if line.startswith(":"):
                continue

            # 3. Parse key: value headers
            if ":" in line:
                field, value = line.split(":", 1)
                if value.startswith(" "):
                    value = value[1:]
                
                if field == "event":
                    current_event_type = value
                elif field == "data":
                    current_data.append(value)
            else:
                # Valueless line is the key itself
                if line == "event":
                    current_event_type = ""
                elif line == "data":
                    current_data.append("")


def load_env_file(path: Path) -> dict[str, str]:
    """
    Reads dynamic local configurations mapping lines KEY=VALUE, 
    stripping whitespaces and handling comment hashes.
    """
    env_dict = {}
    if not path.exists():
        return env_dict

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_stripped = line.strip()
            if not line_stripped or line_stripped.startswith("#"):
                continue

            # State-machine quote parser to strip trailing comments safely
            in_quotes = False
            quote_char = None
            comment_idx = -1

            for i, char in enumerate(line_stripped):
                if char in ('"', "'"):
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                    elif quote_char == char:
                        in_quotes = False
                        quote_char = None
                elif char == '#' and not in_quotes:
                    comment_idx = i
                    break

            # Slices off trailing comments outside quotes
            if comment_idx != -1:
                line_stripped = line_stripped[:comment_idx].strip()

            if not line_stripped or line_stripped.startswith("#"):
                continue

            if "=" not in line_stripped:
                continue

            key, val = line_stripped.split("=", 1)
            key = key.strip()
            val = val.strip()

            # Remove boundary quotes from key if matching
            if len(key) >= 2 and key[0] in ('"', "'") and key[0] == key[-1]:
                key = key[1:-1]

            # Remove boundary quotes from value if matching
            if len(val) >= 2 and val[0] in ('"', "'") and val[0] == val[-1]:
                val = val[1:-1]

            env_dict[key] = val

    return env_dict
