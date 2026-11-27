"""Run the System One decision-model benchmark through pydantic-ai with Logfire tracing.

Reuses the test cases, HTTP helpers, and scoring from benchmarks/head_to_head.py so the
results are directly comparable to the original urllib-based run.
"""

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RequestUsage

BENCH_DIR = Path("/Users/mfranz/tmp/ollama-dms/benchmarks")
HEAD_TO_HEAD = BENCH_DIR / "head_to_head.py"

logfire.configure(service_name="pai-poc-systemone")
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)


def _load_head_to_head() -> Any:
    spec = importlib.util.spec_from_file_location("head_to_head", HEAD_TO_HEAD)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


h2h = _load_head_to_head()

MODELS = h2h.MODELS + ["nimble-v3:9b-nvfp4"]


class SystemOneModel(Model):
    """pydantic-ai Model that proxies a case to Ollama's non-OpenAI /v1/systemone endpoint.

    The user prompt must be a JSON object: {"model": ..., "state": ..., "questions": {...}}.
    The endpoint's answers are returned as the model's text response (JSON).
    """

    def __init__(self, model_name: str):
        super().__init__()
        self._model_name = model_name

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
        case = _last_user_json(messages)
        body = {"model": self._model_name, "state": case["state"], "questions": case["questions"], "keep_alive": "5m"}
        resp, _ms = h2h.http_json("/v1/systemone", body)
        return ModelResponse(
            parts=[TextPart(content=json.dumps(resp.get("answers", {})))],
            usage=RequestUsage(input_tokens=resp.get("usage", {}).get("input_tokens", 0)),
            model_name=self._model_name,
        )


def _last_user_json(messages: list[ModelMessage]) -> dict:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return json.loads(part.content)
    raise ValueError("no user prompt found")


def run_case(model_name: str, case: dict) -> tuple[dict, float]:
    agent: Agent[None, str] = Agent(SystemOneModel(model_name), output_type=str)
    prompt = json.dumps({"state": case["state"], "questions": case["questions"]})
    t0 = time.perf_counter()
    result = agent.run_sync(prompt)
    ms = (time.perf_counter() - t0) * 1000.0
    return json.loads(result.output), ms


def run_model(model_name: str) -> dict:
    with logfire.span("systemone benchmark {model}", model=model_name):
        ps_before, _ = h2h.http_json("/api/ps")
        for loaded in ps_before.get("models", []):
            h2h.unload_model(loaded.get("name", ""))

        _, warmup_ms = h2h.systemone_call(
            model_name,
            "System warmup check.",
            {"warm": {"type": "noul", "instructions": "Is this a warmup check?"}},
            keep_alive="5m",
        )
        resident = h2h.get_resident_info(model_name)

        latencies, cases, correct, total = [], [], 0, 0
        for tc in h2h.TEST_CASES:
            answers, ms = run_case(model_name, tc)
            latencies.append(ms)
            q_results = {}
            for qname, rule in tc["expected"].items():
                ok, detail = h2h.evaluate_answer(answers.get(qname, {}), rule)
                total += 1
                correct += int(ok)
                q_results[qname] = {"correct": ok, "detail": detail, "raw": answers.get(qname, {})}
            cases.append({"id": tc["id"], "latency_ms": round(ms, 2), "questions": q_results})
            print(f"  [{tc['id']}] {ms:7.1f} ms | " + ", ".join(f"{k}:{v['correct']}({v['detail']})" for k, v in q_results.items()))

        h2h.unload_model(model_name)

    return {
        "model": model_name,
        "warmup_ms": round(warmup_ms, 2),
        "resident": resident,
        "median_ms": round(statistics.median(latencies), 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
        "correct": correct,
        "total": total,
        "cases": cases,
    }


def main() -> None:
    models = sys.argv[1:] or MODELS
    results = []
    for model_name in models:
        print(f"\n=== {model_name} (pydantic-ai) ===")
        results.append(run_model(model_name))
    data = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"), "harness": "pydantic-ai + logfire", "models": results}
    out = BENCH_DIR / f"pydantic_ai_systemone_{time.strftime('%Y-%m-%d_%H%M%S')}.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("\n| Model | Resident GB | Warm median ms | Score |")
    print("| --- | ---: | ---: | ---: |")
    for r in results:
        print(f"| {r['model']} | {r['resident']['size_gb']} | {r['median_ms']} | {r['correct']}/{r['total']} |")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
