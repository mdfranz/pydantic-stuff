"""Exact-payload /v1/systemone client that runs every call through a pydantic-ai Agent, traced by Logfire.

pydantic-ai's built-in `SystemOneModel` rewrites questions (object-valued `instructions`) and returns only the
read value per field, which confounds controlled perturbations and calibration. This thin `Model` sends the
payload verbatim and returns the full response (all probabilities), while still producing agent/model spans.
"""

import json
import os
import time
from typing import Any

import httpx
import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

OLLAMA = os.getenv("SYSTEM_ONE_BASE_URL", "http://localhost:11434").rstrip("/")
TIMEOUT = httpx.Timeout(600.0, connect=10.0)

_configured = False


def configure_logfire(service_name: str) -> None:
    global _configured
    if _configured:
        return
    logfire.configure(service_name=service_name)
    logfire.instrument_pydantic_ai()
    logfire.instrument_httpx(capture_all=True)
    _configured = True


class RawSystemOneModel(Model):
    """Sends the user prompt (a JSON `{state, questions, images?}` object) verbatim to /v1/systemone."""

    def __init__(self, model_name: str, *, keep_alive: str | int = "10m"):
        super().__init__()
        self._model_name = model_name
        self._keep_alive = keep_alive

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def system(self) -> str:
        return "ollama-systemone"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        body = {"model": self._model_name, "keep_alive": self._keep_alive, **_last_user_json(messages)}
        # A fresh client per call: Agent.run_sync uses a new event loop each time.
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{OLLAMA}/v1/systemone", json=body)
        try:
            data = resp.json()
        except ValueError:
            data = {"error": resp.text}
        if resp.status_code >= 400 and "error" not in data:
            data = {"error": f"HTTP {resp.status_code}"}
        data["_status"] = resp.status_code
        usage = data.get("usage") or {}
        return ModelResponse(
            parts=[TextPart(content=json.dumps(data))],
            usage=RequestUsage(input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0)),
            model_name=self._model_name,
        )


def _last_user_json(messages: list[ModelMessage]) -> dict[str, Any]:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return json.loads(part.content)
    raise ValueError("no JSON user prompt found")


_agents: dict[str, Agent[None, str]] = {}


def ask(model_name: str, state: Any, questions: dict[str, Any], *, images: list[str] | None = None) -> tuple[dict, float]:
    """One /v1/systemone call through pydantic-ai. Returns (response dict, wall ms). Errors come back as {'error': ...}."""
    agent = _agents.get(model_name)
    if agent is None:
        agent = _agents[model_name] = Agent(RawSystemOneModel(model_name), output_type=str)
    payload: dict[str, Any] = {"state": state, "questions": questions}
    if images:
        payload["images"] = images
    t0 = time.perf_counter()
    try:
        result = agent.run_sync(json.dumps(payload))
        data = json.loads(result.output)
    except Exception as e:  # transport failures, timeouts
        data = {"error": f"{type(e).__name__}: {e}"}
    return data, (time.perf_counter() - t0) * 1000.0


def loaded_models() -> list[dict]:
    return httpx.get(f"{OLLAMA}/api/ps", timeout=30).json().get("models", [])


def unload_all() -> None:
    for m in loaded_models():
        httpx.post(f"{OLLAMA}/api/generate", json={"model": m["name"], "keep_alive": 0}, timeout=60)
    deadline = time.time() + 30
    while loaded_models() and time.time() < deadline:
        time.sleep(0.5)


def resident_info(model_name: str) -> dict:
    for m in loaded_models():
        if model_name in (m.get("name"), m.get("model")):
            return {"size_gb": round(m.get("size", 0) / 1024**3, 2), "context_length": m.get("context_length")}
    return {"size_gb": None, "context_length": None}


def warmup(model_name: str) -> float:
    _, ms = ask(model_name, "Warmup.", {"w": {"type": "noul", "instructions": "Is this a warmup check?"}})
    return ms
