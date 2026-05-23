from __future__ import annotations
import json
import logging
import os
import random
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Iterable, Sequence
from codex.types import PromptRequest, ModelResponse

logger = logging.getLogger("codex")

__all__ = [
    "ModelClient",
    "ModelStreamEvent",
    "OpenAIResponsesModel",
    "RemoteCompactionError",
    "ScriptedResponsesModel",
    "collect_stream_response",
    "iter_model_stream_events"
]


class RemoteCompactionError(Exception):
    pass


class ModelStreamEvent:
    def __init__(self, type: str, payload: dict[str, Any] = None):
        self.type = type
        self.payload = payload if payload is not None else {}

    def __repr__(self) -> str:
        return f"ModelStreamEvent(type={self.type!r}, payload={self.payload!r})"


class ModelClient:
    def __init__(self) -> None:
        self.delegate = ScriptedResponsesModel.from_env()
        if self.delegate is None:
            self.delegate = OpenAIResponsesModel()

    def create(self, request: PromptRequest) -> ModelResponse:
        return self.delegate.create(request)

    def stream(self, request: PromptRequest) -> Iterable[ModelStreamEvent]:
        return self.delegate.stream(request)


class ScriptedResponsesModel:
    def __init__(self, responses: Sequence[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[PromptRequest] = []
        self._response_idx = 0

    @classmethod
    def from_env(cls) -> ScriptedResponsesModel | None:
        fake_val = os.environ.get("PY_CODEX_FAKE_RESPONSES")
        if not fake_val:
            return None
        fake_val = fake_val.strip()
        if fake_val.startswith(("[", "{")):
            try:
                responses = json.loads(fake_val)
                if isinstance(responses, dict):
                    responses = [responses]
                return cls(responses)
            except Exception as e:
                logger.error(f"Failed to parse inline fake responses: {e}")
                return None
        try:
            path = Path(fake_val)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    responses = json.load(f)
                    return cls(responses)
        except Exception as e:
            logger.error(f"Failed to load scripted responses from {fake_val}: {e}")
        return None

    def create(self, request: PromptRequest) -> ModelResponse:
        self.requests.append(request)
        if self._response_idx >= len(self.responses):
            resp = {"id": "scripted_resp", "output": []}
        else:
            resp = self.responses[self._response_idx]
            self._response_idx += 1
            
        return ModelResponse(id=resp.get("id") or "scripted_resp", output=resp.get("output") or [], raw=resp)

    def stream(self, request: PromptRequest) -> Iterable[ModelStreamEvent]:
        self.requests.append(request)
        if self._response_idx >= len(self.responses):
            resp = {"id": "scripted_resp", "output": []}
        else:
            resp = self.responses[self._response_idx]
            self._response_idx += 1
            
        response_id = resp.get("id") or "scripted_resp"
        output_items = resp.get("output") or []
        
        yield ModelStreamEvent(type="response.created", payload={"response_id": response_id, "id": response_id})
        yield ModelStreamEvent(type="created", payload={"response_id": response_id, "id": response_id})
        
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "message":
                role = item.get("role", "assistant")
                content = item.get("content", [])
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        text = c.get("text", "")
                        yield ModelStreamEvent(type="response.output_text.delta", payload={"delta": text})
                        yield ModelStreamEvent(type="output_text_delta", payload={"delta": text})
                        
            yield ModelStreamEvent(type="response.output_item.done", payload={"item": item})
            yield ModelStreamEvent(type="output_item_done", payload={"item": item})
            
        yield ModelStreamEvent(type="response.completed", payload={"response_id": response_id, "id": response_id, "usage": resp.get("usage")})
        yield ModelStreamEvent(type="completed", payload={"response_id": response_id, "id": response_id, "usage": resp.get("usage")})


class OpenAIResponsesModel:
    def __init__(self) -> None:
        pass

    def _get_api_key(self) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            # We raise an error that tells the caller to set the key
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return api_key

    def _get_base_url(self, model_slug: str) -> str:
        from codex.types import get_model_preset
        preset = get_model_preset(model_slug)
        # Try to read base_url from model preset model providers config
        # By default, falls back to standard OpenAI endpoint
        return "https://api.openai.com/v1"

    def create(self, request: PromptRequest) -> ModelResponse:
        api_key = self._get_api_key()
        base_url = self._get_base_url(request.model)
        
        payload = request.to_responses_kwargs()
        payload["stream"] = False
        
        req = urllib.request.Request(
            url=f"{base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            method="POST"
        )
        
        # Standard retries for POST requests
        max_attempts = 4
        delay = 0.25
        attempt = 0
        while True:
            attempt += 1
            try:
                with urllib.request.urlopen(req, timeout=300) as response:
                    res_body = response.read().decode("utf-8")
                    data = json.loads(res_body)
                    return ModelResponse(id=data.get("id") or "responses_id", output=data.get("output") or [], raw=data)
            except Exception as e:
                if attempt >= max_attempts:
                    raise
                # Exponential backoff with jitter
                sleep_for = delay + random.uniform(-delay * 0.2, delay * 0.2)
                time.sleep(max(0.01, sleep_for))
                delay = min(2.0, delay * 2.0)

    def stream(self, request: PromptRequest) -> Iterable[ModelStreamEvent]:
        api_key = self._get_api_key()
        base_url = self._get_base_url(request.model)
        
        payload = request.to_responses_kwargs()
        payload["stream"] = True
        
        req = urllib.request.Request(
            url=f"{base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "Accept": "text/event-stream",
            },
            method="POST"
        )
        
        # Stream retry parameters (from config: 10 max retries, 200ms delay base)
        max_attempts = 10
        attempt = 0
        delay = 0.2
        
        while True:
            attempt += 1
            try:
                # We open the stream and read line-by-line in real-time
                with urllib.request.urlopen(req, timeout=300) as response:
                    buffer = ""
                    for chunk in response:
                        buffer += chunk.decode("utf-8")
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line = line.strip()
                            if not line:
                                continue
                            if line.startswith("data:"):
                                line = line[5:].strip()
                            if line == "[DONE]":
                                return
                            try:
                                data = json.loads(line)
                                yield ModelStreamEvent(type=data.get("type", ""), payload=data.get("payload", data))
                            except Exception:
                                pass
                # Stream successfully completed
                return
            except Exception as e:
                # In standard Responses API: if stream drop occurs, we retry using exponential backoff
                if attempt >= max_attempts:
                    raise
                logger.warning(f"SSE stream drop detected, retrying (attempt {attempt}/{max_attempts}): {e}")
                sleep_for = delay + random.uniform(-delay * 0.2, delay * 0.2)
                time.sleep(max(0.01, sleep_for))
                delay = min(5.0, delay * 2.0)

    def compact(
        self,
        request: PromptRequest,
        *,
        session_id: str | None = None,
        thread_id: str | None = None,
        installation_id: str | None = None
    ) -> list[dict[str, Any]]:
        api_key = self._get_api_key()
        base_url = self._get_base_url(request.model)
        
        payload = request.to_compact_payload()
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        }
        if thread_id is not None:
            headers["x-client-request-id"] = thread_id
            headers["x-codex-thread-id"] = thread_id
        if session_id is not None:
            headers["x-codex-session-id"] = session_id
        if installation_id is not None:
            headers["x-codex-installation-id"] = installation_id
            
        req = urllib.request.Request(
            url=f"{base_url}/responses/compact",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        
        max_attempts = 4
        delay = 0.25
        attempt = 0
        while True:
            attempt += 1
            try:
                with urllib.request.urlopen(req, timeout=300) as response:
                    res_body = response.read().decode("utf-8")
                    data = json.loads(res_body)
                    # Returns list[dict] representing the compacted output history items!
                    return data.get("output") or []
            except Exception as e:
                if attempt >= max_attempts:
                    raise RemoteCompactionError(f"Remote compaction failed: {e}")
                sleep_for = delay + random.uniform(-delay * 0.2, delay * 0.2)
                time.sleep(max(0.01, sleep_for))
                delay = min(2.0, delay * 2.0)


def collect_stream_response(events: Iterable[Any]) -> ModelResponse:
    response_id = ""
    output_items = []
    raw_events = []
    
    for event in events:
        if hasattr(event, "type") and hasattr(event, "payload"):
            evt_type = event.type
            evt_payload = event.payload
        elif isinstance(event, dict):
            evt_type = event.get("type", "")
            evt_payload = event.get("payload", event)
        else:
            continue
            
        raw_events.append({"type": evt_type, "payload": evt_payload})
        
        # Match standard or nested response completes
        if evt_type in ("completed", "response.completed"):
            response_id = evt_payload.get("response_id") or evt_payload.get("id") or response_id
            
        if evt_type in ("output_item_done", "response.output_item.done"):
            item = evt_payload.get("item")
            if item is not None:
                output_items.append(item)
                
    return ModelResponse(id=response_id, output=output_items, raw={"events": raw_events})


def iter_model_stream_events(events: Iterable[Any]) -> Iterable[ModelStreamEvent]:
    for event in events:
        if isinstance(event, str):
            line = event.strip()
            if line.startswith("data:"):
                line = line[5:].strip()
            if not line or line == "[DONE]":
                continue
            try:
                data = json.loads(line)
                yield ModelStreamEvent(type=data.get("type", ""), payload=data.get("payload", data))
            except Exception:
                pass
        elif isinstance(event, dict):
            yield ModelStreamEvent(type=event.get("type", ""), payload=event.get("payload", event))
        elif hasattr(event, "type") and hasattr(event, "payload"):
            yield ModelStreamEvent(type=event.type, payload=event.payload)


def load_env_file(path: Path) -> dict[str, str]:
    env = {}
    if not path.exists():
        return env
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip()
                    if len(v) >= 2 and ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
                        v = v[1:-1]
                    env[k] = v
    except Exception as e:
        logger.error(f"Failed to load env file at {path}: {e}")
    return env
