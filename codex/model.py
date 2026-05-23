from __future__ import annotations
import json
import os
import urllib.request
import urllib.error
import uuid
import ssl
try:
    import certifi
    _has_certifi = True
except ImportError:
    _has_certifi = False
from pathlib import Path
from typing import Any, Iterable, Sequence
from dataclasses import dataclass, field
from codex.types import PromptRequest, ModelResponse

def get_ssl_context() -> ssl.SSLContext | None:
    if os.environ.get("CODEX_SSL_NO_VERIFY") == "true":
        try:
            return ssl._create_unverified_context()
        except Exception:
            pass
    if _has_certifi:
        try:
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            pass
    try:
        return ssl.create_default_context()
    except Exception:
        return None

# --- Model Stream Event ------------------------------------------------------
@dataclass
class ModelStreamEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)

# --- Errors ------------------------------------------------------------------
class RemoteCompactionError(Exception):
    pass

# --- Abstract Base Model Client ----------------------------------------------
class ModelClient:
    def create(self, request: PromptRequest) -> ModelResponse:
        raise NotImplementedError("create must be implemented")

    def stream(self, request: PromptRequest) -> Iterable[ModelStreamEvent]:
        raise NotImplementedError("stream must be implemented")

# --- Standard Library SSE / HTTP responses model -----------------------------
def get_openai_base_url() -> str:
    base = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    if base:
        return base.rstrip("/")
    return "https://api.openai.com"

def map_openai_sse_event_to_model_stream_event(openai_event: dict[str, Any]) -> ModelStreamEvent | None:
    ev_type = openai_event.get("type")
    if ev_type == "response.created":
        return ModelStreamEvent("turn.started", {})
    elif ev_type == "response.output_item.added":
        item = openai_event.get("item", {})
        return ModelStreamEvent("item.started", {
            "item_id": item.get("id"),
            "type": item.get("type"),
            "name": item.get("name"),
            "call_id": item.get("call_id") or item.get("id"),
        })
    elif ev_type == "response.output_item.delta":
        delta_text = ""
        if "delta" in openai_event:
            delta_val = openai_event["delta"]
            if isinstance(delta_val, str):
                delta_text = delta_val
            elif isinstance(delta_val, dict):
                delta_text = delta_val.get("arguments", "") or delta_val.get("text", "")
        return ModelStreamEvent("item.delta", {
            "item_id": openai_event.get("item_id"),
            "delta": delta_text,
        })
    elif ev_type == "response.output_item.done":
        item = openai_event.get("item", {})
        return ModelStreamEvent("item.completed", {
            "item_id": item.get("id"),
            "item": item,
        })
    elif ev_type == "response.done":
        resp = openai_event.get("response", {})
        usage = resp.get("usage", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        return ModelStreamEvent("turn.completed", {
            "usage": usage,
        })
    return None

class OpenAIResponsesModel(ModelClient):
    def create(self, request: PromptRequest) -> ModelResponse:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
            
        kwargs = request.to_responses_kwargs()
        kwargs["stream"] = False
        kwargs["store"] = False
        
        base_url = get_openai_base_url()
        url = f"{base_url}/v1/responses"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        installation_id = kwargs.get("client_metadata", {}).get("x-codex-installation-id")
        if installation_id:
            headers["x-codex-installation-id"] = installation_id
            
        data = json.dumps(kwargs).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, context=get_ssl_context()) as resp:
                res_body = resp.read().decode("utf-8")
                res_json = json.loads(res_body)
                return ModelResponse(
                    id=res_json.get("id", "resp-unknown"),
                    output=res_json.get("output", []),
                    raw=res_json,
                )
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API returned HTTP {e.code}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to call OpenAI API: {e}") from e

    def stream(self, request: PromptRequest) -> Iterable[ModelStreamEvent]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
            
        kwargs = request.to_responses_kwargs()
        kwargs["stream"] = True
        kwargs["store"] = False
        
        base_url = get_openai_base_url()
        url = f"{base_url}/v1/responses"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        installation_id = kwargs.get("client_metadata", {}).get("x-codex-installation-id")
        if installation_id:
            headers["x-codex-installation-id"] = installation_id
            
        data = json.dumps(kwargs).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        try:
            resp = urllib.request.urlopen(req, context=get_ssl_context())
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI API returned HTTP {e.code}: {err_body}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to call OpenAI API: {e}") from e
            
        buffer = ""
        try:
            with resp:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="replace")
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line.startswith("data:"):
                            data_part = line[5:].strip()
                            if data_part == "[DONE]":
                                break
                            try:
                                event_data = json.loads(data_part)
                                mapped = map_openai_sse_event_to_model_stream_event(event_data)
                                if mapped:
                                    yield mapped
                            except Exception:
                                pass
        finally:
            pass

    def compact(
        self,
        request: PromptRequest,
        *,
        session_id: str | None = None,
        thread_id: str | None = None,
        installation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
            
        payload = request.to_compact_payload()
        
        base_url = get_openai_base_url()
        url = f"{base_url}/v1/responses/compact"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if installation_id:
            headers["x-codex-installation-id"] = installation_id
        if session_id:
            headers["x-codex-session-id"] = session_id
        if thread_id:
            headers["x-codex-thread-id"] = thread_id
            
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        
        try:
            with urllib.request.urlopen(req, context=get_ssl_context()) as resp:
                res_body = resp.read().decode("utf-8")
                res_json = json.loads(res_body)
                if "output" in res_json and isinstance(res_json["output"], list):
                    return res_json["output"]
                raise RemoteCompactionError(f"Unexpected compaction response body: {res_body}")
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RemoteCompactionError(f"Remote /responses/compact request failed: {e.code}: {err_body}") from e
        except Exception as e:
            raise RemoteCompactionError(f"Failed to connect to remote compaction endpoint: {e}") from e

# --- Scripted/Mock responses model (Testing-only) ---------------------------
class ScriptedResponsesModel(ModelClient):
    def __init__(self, responses: Sequence[dict[str, Any]]):
        self.responses = list(responses)
        self.requests: list[PromptRequest] = []

    def create(self, request: PromptRequest) -> ModelResponse:
        self.requests.append(request)
        if not self.responses:
            raise RuntimeError("No scripted responses available.")
        resp = self.responses.pop(0)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(resp["error"])
            
        resp_id = resp.get("id") or resp.get("item_id") or resp.get("call_id") or str(uuid.uuid4())
        output = resp.get("output", [])
        return ModelResponse(id=resp_id, output=output, raw=resp)

    def stream(self, request: PromptRequest) -> Iterable[ModelStreamEvent]:
        self.requests.append(request)
        if not self.responses:
            raise RuntimeError("No scripted responses available.")
        resp = self.responses.pop(0)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(resp["error"])
            
        if "stream" in resp and isinstance(resp["stream"], list):
            for ev in resp["stream"]:
                yield ModelStreamEvent(type=ev["type"], payload=ev["payload"])
        else:
            yield ModelStreamEvent("turn.started", {})
            output = resp.get("output", [])
            last_item_id = "mock-resp"
            for item in output:
                item_id = item.get("call_id") or item.get("id") or str(uuid.uuid4())
                last_item_id = item_id
                item_type = item.get("type", "message")
                
                if item_type in ("function_call", "custom_tool_call"):
                    yield ModelStreamEvent("item.started", {
                        "item_id": item_id,
                        "type": item_type,
                        "name": item.get("name"),
                        "call_id": item_id,
                    })
                    delta_text = item.get("arguments") or item.get("input") or ""
                    yield ModelStreamEvent("item.delta", {"item_id": item_id, "delta": delta_text})
                    yield ModelStreamEvent("item.completed", {"item_id": item_id, "item": item})
                elif item_type == "message":
                    yield ModelStreamEvent("item.started", {
                        "item_id": item_id,
                        "type": "message",
                        "role": item.get("role", "assistant"),
                    })
                    for part in item.get("content", []):
                        if isinstance(part, dict) and "text" in part:
                            yield ModelStreamEvent("item.delta", {"item_id": item_id, "delta": part["text"]})
                    yield ModelStreamEvent("item.completed", {"item_id": item_id, "item": item})
                else:
                    yield ModelStreamEvent("item.started", {
                        "item_id": item_id,
                        "type": item_type,
                        "call_id": item_id,
                    })
                    yield ModelStreamEvent("item.completed", {"item_id": item_id, "item": item})
                    
            yield ModelStreamEvent("model.response", {"response_id": last_item_id})
            
            usage = resp.get("usage") or {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15, "reasoning_output_tokens": 0}
            yield ModelStreamEvent("turn.completed", {"usage": usage})

    @classmethod
    def from_env(cls) -> ScriptedResponsesModel | None:
        fake_env = os.environ.get("PY_CODEX_FAKE_RESPONSES")
        if not fake_env:
            return None
        try:
            responses = json.loads(fake_env)
            if isinstance(responses, dict):
                responses = [responses]
            return cls(responses)
        except Exception:
            return cls([])

# --- Env file loader --------------------------------------------------------
def load_env_file(path: Path) -> dict[str, str]:
    res = {}
    path = Path(path)
    if not path.exists():
        return res
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                        val = val[1:-1]
                    res[key] = val
    except Exception:
        pass
    return res

# --- Default Model Client selection ------------------------------------------
def default_model_client() -> ModelClient:
    fake_env = os.environ.get("PY_CODEX_FAKE_RESPONSES")
    if fake_env:
        client = ScriptedResponsesModel.from_env()
        if client is not None:
            return client
    return OpenAIResponsesModel()

# --- Stream processing helpers ----------------------------------------------
def iter_model_stream_events(events: Iterable[Any]) -> Iterable[ModelStreamEvent]:
    for ev in events:
        if isinstance(ev, ModelStreamEvent):
            yield ev
        elif isinstance(ev, dict):
            payload = ev.get("payload") if "payload" in ev else ev
            yield ModelStreamEvent(type=ev.get("type", "unknown"), payload=payload)
        elif hasattr(ev, "type") and hasattr(ev, "payload"):
            yield ModelStreamEvent(type=getattr(ev, "type"), payload=getattr(ev, "payload"))

def collect_stream_response(events: Iterable[Any]) -> ModelResponse:
    response_id = "routed"
    items_by_id = {}
    outputs = []
    raw_data = []
    
    for ev in iter_model_stream_events(events):
        raw_data.append(ev.payload)
        
        # Determine ev_type and raw payload
        if hasattr(ev, "type") and hasattr(ev, "payload"):
            ev_type = ev.type
            payload = ev.payload
        elif isinstance(ev, dict):
            ev_type = ev.get("type")
            payload = ev.get("payload") if "payload" in ev else ev
        else:
            ev_type = getattr(ev, "type", "unknown")
            payload = getattr(ev, "payload", {})
            
        if ev_type in ("response.completed", "response.done", "model.response"):
            resp_data = payload.get("response") if isinstance(payload, dict) else None
            if resp_data and isinstance(resp_data, dict):
                response_id = resp_data.get("id") or response_id
                if "output" in resp_data and isinstance(resp_data["output"], list):
                    outputs = list(resp_data["output"])
            elif isinstance(payload, dict) and "response_id" in payload:
                response_id = payload["response_id"]
                
        elif ev_type in ("response.output_item.added", "item.started"):
            item = payload.get("item") if isinstance(payload, dict) else None
            if not item:
                item = payload
            if isinstance(item, dict):
                item_id = item.get("id") or item.get("item_id") or item.get("call_id")
                if item_id:
                    item_type = item.get("type") or "message"
                    # ensure correct default schema keys
                    if item_type in ("function_call", "custom_tool_call"):
                        if "arguments" not in item:
                            item["arguments"] = ""
                        if "input" not in item:
                            item["input"] = ""
                    elif item_type == "message" and "content" not in item:
                        item["content"] = [{"type": "output_text", "text": ""}]
                    items_by_id[item_id] = item
                    
        elif ev_type in ("response.output_item.delta", "item.delta"):
            if isinstance(payload, dict):
                item_id = payload.get("item_id") or payload.get("id")
                delta_val = payload.get("delta")
                delta_text = ""
                if isinstance(delta_val, str):
                    delta_text = delta_val
                elif isinstance(delta_val, dict):
                    delta_text = delta_val.get("arguments", "") or delta_val.get("text", "")
                else:
                    delta_text = payload.get("text", "") or ""
                    
                if item_id and item_id in items_by_id:
                    item = items_by_id[item_id]
                    item_type = item.get("type")
                    if item_type == "function_call":
                        item["arguments"] += delta_text
                    elif item_type == "custom_tool_call":
                        item["input"] += delta_text
                    elif item_type == "message":
                        if "content" in item and item["content"]:
                            item["content"][0]["text"] += delta_text
                            
        elif ev_type in ("response.output_item.done", "item.completed"):
            item = payload.get("item") if isinstance(payload, dict) else None
            if not item:
                item = payload
            if isinstance(item, dict):
                item_id = item.get("id") or item.get("item_id") or item.get("call_id")
                if not item_id and items_by_id:
                    # fallback to only active
                    item_id = list(items_by_id.keys())[-1]
                if item_id:
                    items_by_id[item_id] = item
                    if response_id == "routed":
                        response_id = item_id
                if item not in outputs:
                    outputs.append(item)
                        
    if not outputs and items_by_id:
        outputs = list(items_by_id.values())
        
    return ModelResponse(id=response_id, output=outputs, raw={"events": raw_data})
