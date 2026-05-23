from __future__ import annotations
import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Iterable, Sequence

from codex.types import PromptRequest, ModelResponse


class ModelStreamEvent:
    def __init__(self, type: str, payload: dict[str, Any]) -> None:
        self.type = type
        self.payload = payload


class ModelClient:
    def __init__(self) -> None:
        pass

    def create(self, request: PromptRequest) -> ModelResponse:
        return ModelResponse(id="stub", output=[])

    def stream(self, request: PromptRequest) -> Iterable[ModelStreamEvent]:
        return []


class OpenAIResponsesModel:
    def __init__(self) -> None:
        pass

    def compact(
        self,
        request: PromptRequest,
        *,
        session_id: str | None = None,
        thread_id: str | None = None,
        installation_id: str | None = None
    ) -> list[dict[str, Any]]:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        endpoint = f"{base_url.rstrip('/')}/responses/compact"
        
        payload = request.to_compact_payload()
        data_bytes = json.dumps(payload).encode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "responses_websockets=2026-02-06"
        }
        if installation_id:
            headers["x-codex-installation-id"] = installation_id
        if session_id:
            headers["x-codex-session-id"] = session_id
        if thread_id:
            headers["x-codex-thread-id"] = thread_id
            
        req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
        try:
            timeout_sec = float(os.environ.get("CODEX_TIMEOUT_SEC") or "60.0")
        except ValueError:
            timeout_sec = 60.0

        try:
            response = urllib.request.urlopen(req, timeout=timeout_sec)
            body_bytes = response.read()
            response.close()
            res_json = json.loads(body_bytes.decode("utf-8"))
            if isinstance(res_json, list):
                return res_json
            elif isinstance(res_json, dict) and "output" in res_json:
                return res_json["output"]
            return []
        except urllib.error.HTTPError as err:
            body_bytes = err.read()
            try:
                body = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                body = {}
            message = body.get("error", {}).get("message") or f"HTTP compaction error: {err.code}"
            raise RemoteCompactionError(message)
        except Exception as exc:
            raise RemoteCompactionError(f"Remote compaction connection failure: {exc}")

    def create(self, request: PromptRequest) -> ModelResponse:
        stream = self.stream(request)
        return collect_stream_response(stream)

    def stream(self, request: PromptRequest) -> Iterable[ModelStreamEvent]:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        endpoint = f"{base_url.rstrip('/')}/responses"
        
        payload_dict = request.to_responses_kwargs()
        payload_dict["stream"] = True
        data_bytes = json.dumps(payload_dict).encode("utf-8")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "responses_websockets=2026-02-06"
        }
        if request.client_metadata:
            for k, v in request.client_metadata.items():
                headers[k] = v
                
        req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")
        try:
            timeout_sec = float(os.environ.get("CODEX_TIMEOUT_SEC") or "60.0")
        except ValueError:
            timeout_sec = 60.0

        try:
            response = urllib.request.urlopen(req, timeout=timeout_sec)
        except urllib.error.HTTPError as err:
            body_bytes = err.read()
            try:
                body = json.loads(body_bytes.decode("utf-8"))
            except Exception:
                body = {}
                
            import openai
            status_code = err.code
            message = body.get("error", {}).get("message") or str(err)
            
            if status_code == 400:
                raise openai.BadRequestError(message, response=err, body=body)
            elif status_code == 401:
                raise openai.AuthenticationError(message, response=err, body=body)
            elif status_code == 403:
                raise openai.PermissionDeniedError(message, response=err, body=body)
            elif status_code == 404:
                raise openai.NotFoundError(message, response=err, body=body)
            elif status_code == 429:
                raise openai.RateLimitError(message, response=err, body=body)
            elif status_code >= 500:
                raise openai.InternalServerError(message, response=err, body=body)
            else:
                raise openai.APIStatusError(message, response=err, body=body)
        except Exception as exc:
            import openai
            raise openai.APIConnectionError(message=f"Connection failure: {exc}", request=req)
            
        try:
            buffer = b""
            while True:
                chunk = response.read(1024)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode("utf-8").strip()
                    if line_str.startswith("data:"):
                        data_content = line_str[5:].strip()
                        if data_content == "[DONE]":
                            break
                        try:
                            event_dict = json.loads(data_content)
                            for model_event in iter_model_stream_events([event_dict]):
                                yield model_event
                        except Exception as exc:
                            import logging; logging.warning(f"Swallowed exception trace: {exc}")
        finally:
            response.close()


class RemoteCompactionError(Exception):
    pass


class ScriptedResponsesModel:
    def __init__(self, responses: Sequence[dict[str, Any]]) -> None:
        self.responses = responses
        self.requests: list[PromptRequest] = []
        self.current_turn = 0

    @classmethod
    def from_env(cls) -> ScriptedResponsesModel | None:
        fake_resp_path = os.environ.get("PY_CODEX_FAKE_RESPONSES")
        if not fake_resp_path:
            return None
        try:
            path = Path(fake_resp_path)
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return cls(data)
        except Exception as exc:
            import logging; logging.warning(f"Swallowed exception trace: {exc}")
        return None

    def compact(
        self,
        request: PromptRequest,
        *,
        session_id: str | None = None,
        thread_id: str | None = None,
        installation_id: str | None = None
    ) -> list[dict[str, Any]]:
        self.requests.append(request)
        if self.current_turn < len(self.responses):
            turn_events = self.responses[self.current_turn]
            self.current_turn += 1
            resp = collect_stream_response(turn_events)
            return resp.output
        return []

    def create(self, request: PromptRequest) -> ModelResponse:
        self.requests.append(request)
        if self.current_turn < len(self.responses):
            turn_events = self.responses[self.current_turn]
            self.current_turn += 1
            return collect_stream_response(turn_events)
        return ModelResponse(id="resp-default", output=[])

    def stream(self, request: PromptRequest) -> Iterable[ModelStreamEvent]:
        self.requests.append(request)
        if self.current_turn < len(self.responses):
            turn_events = self.responses[self.current_turn]
            self.current_turn += 1
            return iter_model_stream_events(turn_events)
        return []


def collect_stream_response(events: Iterable[Any]) -> ModelResponse:
    response_id = "resp-default"
    output_items = []
    usage_stats = None
    
    for ev in events:
        if isinstance(ev, ModelStreamEvent):
            ev_type = ev.type
            ev_payload = ev.payload
        elif isinstance(ev, dict):
            ev_type = ev.get("type")
            ev_payload = ev.get("payload") or ev
        else:
            ev_type = getattr(ev, "type", None)
            ev_payload = getattr(ev, "payload", None) or ev
            
        if not ev_type:
            continue
            
        if ev_type == "response.created":
            rid = ev_payload.get("id") or ev_payload.get("response", {}).get("id")
            if rid:
                response_id = rid
        elif ev_type == "response.output_item.done":
            item = ev_payload.get("item")
            if item:
                output_items.append(item)
        elif ev_type in ("response.done", "response.completed"):
            rid = ev_payload.get("id") or ev_payload.get("response", {}).get("id")
            if rid:
                response_id = rid
            usage = ev_payload.get("usage") or ev_payload.get("response", {}).get("usage")
            if usage:
                usage_stats = usage
                
    raw = {
        "id": response_id,
        "object": "response",
        "output": output_items
    }
    if usage_stats:
        raw["usage"] = usage_stats
        
    return ModelResponse(id=response_id, output=output_items, raw=raw)


def iter_model_stream_events(events: Iterable[Any]) -> Iterable[ModelStreamEvent]:
    res = []
    for ev in events:
        if isinstance(ev, ModelStreamEvent):
            res.append(ev)
            continue
            
        if not isinstance(ev, dict):
            continue
            
        ev_type = ev.get("type")
        if not ev_type:
            continue
            
        if ev_type == "response.failed":
            import openai
            resp_val = ev.get("response", {})
            error = resp_val.get("error", {})
            message = error.get("message", "response.failed event received")
            code = error.get("code")
            
            from unittest.mock import MagicMock
            mock_response = MagicMock()
            mock_response.status_code = 400
            
            if code == "context_length_exceeded" or "context length" in message.lower():
                raise openai.BadRequestError(
                    message=message,
                    response=mock_response,
                    body={"error": {"code": "context_length_exceeded"}}
                )
            elif code == "invalid_api_key" or "api key" in message.lower():
                mock_response.status_code = 401
                raise openai.AuthenticationError(
                    message=message,
                    response=mock_response,
                    body={"error": {"code": "invalid_api_key"}}
                )
            elif code == "rate_limit_exceeded" or "rate limit" in message.lower():
                mock_response.status_code = 429
                raise openai.RateLimitError(
                    message=message,
                    response=mock_response,
                    body={"error": {"code": "rate_limit_exceeded"}}
                )
            else:
                raise openai.APIStatusError(message, response=mock_response, body=error)
                
        if ev_type == "response.incomplete":
            import openai
            resp_val = ev.get("response", {})
            reason = resp_val.get("incomplete_details", {}).get("reason", "unknown")
            message = f"Incomplete response returned, reason: {reason}"
            from unittest.mock import MagicMock
            mock_response = MagicMock()
            mock_response.status_code = 400
            raise openai.BadRequestError(message, response=mock_response, body=resp_val)
            
        payload = {}
        if ev_type == "response.created":
            payload = {"id": ev.get("response", {}).get("id")}
        elif ev_type == "response.output_item.done":
            payload = {"item": ev.get("item")}
        elif ev_type in ("response.done", "response.completed"):
            payload = {
                "id": ev.get("response", {}).get("id"),
                "usage": ev.get("response", {}).get("usage"),
                "end_turn": ev.get("response", {}).get("end_turn")
            }
        elif ev_type == "response.output_text.delta":
            payload = {"delta": ev.get("delta")}
        elif ev_type == "response.custom_tool_call_input.delta":
            payload = {
                "delta": ev.get("delta"),
                "item_id": ev.get("item_id") or ev.get("call_id"),
                "call_id": ev.get("call_id")
            }
        elif ev_type == "response.reasoning_summary_text.delta":
            payload = {
                "delta": ev.get("delta"),
                "summary_index": ev.get("summary_index")
            }
        elif ev_type == "response.reasoning_text.delta":
            payload = {
                "delta": ev.get("delta"),
                "content_index": ev.get("content_index")
            }
        elif ev_type == "response.output_item.added":
            payload = {"item": ev.get("item")}
        elif ev_type == "response.reasoning_summary_part.added":
            payload = {"summary_index": ev.get("summary_index")}
        else:
            payload = {k: v for k, v in ev.items() if k != "type"}
            
        res.append(ModelStreamEvent(ev_type, payload))
    return res


def load_env_file(path: Path) -> dict[str, str]:
    res = {}
    if not path.is_file():
        return res
        
    try:
        content = path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line_str = line.strip()
            if not line_str or line_str.startswith('#'):
                continue
                
            if '=' not in line_str:
                continue
                
            key, val = line_str.split('=', 1)
            key = key.strip()
            val = val.strip()
            
            if key.upper().startswith("CODEX_"):
                continue
                
            if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
                
            os.environ[key] = val
            res[key] = val
    except Exception as exc:
        import logging; logging.warning(f"Swallowed exception trace: {exc}")
        
    return res
