#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic-ai-slim[openai]",
# ]
# ///
"""Head-to-head pydantic-ai + Ollama benchmark for <10B generative decision models on Mac Mini M4."""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput, ToolOutput
from pydantic_ai.providers.ollama import OllamaProvider

OLLAMA_BASE = "http://localhost:11434"
OLLAMA_V1 = f"{OLLAMA_BASE}/v1"

MODELS = [
    "lfm3:1.7b-q8_0",
    "nemotron-3-nano:4b-q4_K_M",
    "granite4.2:8b-q4_K_M",
    "mimo-v2.6:9b-q4_K_M",
    "ornith-1.5:9b",
    "qwen3.5:9b",
    "gemma4:e4b",
]


class TicketDecision(BaseModel):
    """Structured triage decision for a customer support ticket."""

    team: Literal["billing", "shipping", "technical", "account"]
    urgent: bool
    severity: int = Field(ge=0, le=3)
    summary: str = Field(min_length=5)


class RefundVerdict(BaseModel):
    """Final refund decision after inspecting order status."""

    verdict: Literal["approve_refund", "deny_refund", "escalate"]
    order_id: str
    reason: str = Field(min_length=5)


def http_json(path: str, payload: dict | None = None, timeout: int = 180) -> tuple[dict, float]:
    """Send an HTTP GET or POST JSON request to Ollama and return (data, elapsed_ms)."""
    url = f"{OLLAMA_BASE}{path}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw), (time.perf_counter() - t0) * 1000.0


def unload_all_models() -> None:
    """Unload any resident models from Ollama memory via keep_alive=0."""
    ps, _ = http_json("/api/ps")
    for entry in ps.get("models", []):
        name = entry.get("name") or entry.get("model")
        if name:
            try:
                http_json("/api/generate", {"model": name, "prompt": "", "keep_alive": 0}, timeout=30)
            except Exception:
                pass
    if ps.get("models"):
        time.sleep(2.0)


def get_resident_info(model: str) -> dict:
    """Inspect /api/ps for resident GB and active context length."""
    ps, _ = http_json("/api/ps")
    for entry in ps.get("models", []):
        if entry.get("name") == model or entry.get("model") == model:
            return {
                "size_gb": round(entry.get("size", 0) / (1024**3), 2),
                "size_vram_gb": round(entry.get("size_vram", 0) / (1024**3), 2),
                "context_length": entry.get("context_length"),
            }
    return {"size_gb": None, "size_vram_gb": None, "context_length": None}


def probe_speed_and_think(model: str) -> dict:
    """Run /api/chat with think=false to measure M4 tok/s, resident RAM, and think-toggle behavior."""
    # Warmup first so model load time does not skew decode speed
    http_json(
        "/api/chat",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Say ok."}],
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.0, "num_predict": 32},
        },
    )
    res_info = get_resident_info(model)

    resp, elapsed_ms = http_json(
        "/api/chat",
        {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with one concise sentence explaining what a duplicate credit card charge is."}],
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0.0, "num_predict": 256},
        },
    )
    eval_count = resp.get("eval_count", 0) or 0
    eval_ns = resp.get("eval_duration", 0) or 0
    prompt_count = resp.get("prompt_eval_count", 0) or 0
    prompt_ns = resp.get("prompt_eval_duration", 0) or 0
    tok_s = round(eval_count / (eval_ns / 1e9), 1) if eval_ns > 0 else None
    prefill_tok_s = round(prompt_count / (prompt_ns / 1e9), 1) if prompt_ns > 0 else None

    msg = resp.get("message", {})
    thinking_text = msg.get("thinking") or ""
    content_text = msg.get("content") or ""
    has_think_tags = "<think>" in content_text or bool(thinking_text.strip())
    return {
        "resident": res_info,
        "elapsed_ms": round(elapsed_ms, 1),
        "decode_tok_s": tok_s,
        "prefill_tok_s": prefill_tok_s,
        "eval_tokens": eval_count,
        "think_false_respected": not has_think_tags,
        "thinking_chars": len(thinking_text),
    }


def make_model(model_name: str) -> OllamaModel:
    """Create a pydantic-ai OllamaModel pointed at local Ollama /v1."""
    return OllamaModel(
        model_name,
        provider=OllamaProvider(base_url=OLLAMA_V1),
    )


def has_thinking_part(result) -> bool:
    """Check whether any message in a pydantic-ai run contains a ThinkingPart."""
    for msg in result.all_messages():
        for part in getattr(msg, "parts", []):
            if type(part).__name__ == "ThinkingPart" and getattr(part, "content", "").strip():
                return True
    return False


def run_native_output_test(model_name: str) -> dict:
    """Test pydantic-ai NativeOutput (json_schema) on support ticket triage."""
    prompt = (
        "Triage this customer support ticket: "
        "'My credit card was charged twice for order #8821 today and my account is overdrawn. "
        "Please reverse the duplicate charge immediately.'"
    )
    t0 = time.perf_counter()
    try:
        agent = Agent(
            make_model(model_name),
            output_type=NativeOutput(TicketDecision),
            retries=1,
            instructions="Classify the ticket accurately. Respond concisely.",
        )
        res = agent.run_sync(prompt, model_settings={"temperature": 0.0, "max_tokens": 2048})
        ms = (time.perf_counter() - t0) * 1000.0
        out: TicketDecision = res.output
        ok = out.team == "billing" and out.urgent is True and out.severity >= 1 and bool(out.summary.strip())
        usage = res.usage()
        return {
            "pass": ok,
            "latency_ms": round(ms, 1),
            "output_tokens": usage.output_tokens,
            "thinking_part": has_thinking_part(res),
            "output": out.model_dump(),
            "error": None if ok else f"Unexpected values: {out.model_dump()}",
        }
    except Exception as exc:
        return {
            "pass": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "output_tokens": None,
            "thinking_part": None,
            "output": None,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def run_tool_output_test(model_name: str) -> dict:
    """Test pydantic-ai default ToolOutput on the same TicketDecision schema."""
    prompt = (
        "Triage this customer support ticket: "
        "'My credit card was charged twice for order #8821 today and my account is overdrawn. "
        "Please reverse the duplicate charge immediately.'"
    )
    t0 = time.perf_counter()
    try:
        agent = Agent(
            make_model(model_name),
            output_type=ToolOutput(TicketDecision),
            retries=1,
            instructions="Classify the ticket accurately.",
        )
        res = agent.run_sync(prompt, model_settings={"temperature": 0.0, "max_tokens": 2048})
        ms = (time.perf_counter() - t0) * 1000.0
        out: TicketDecision = res.output
        ok = out.team == "billing" and out.urgent is True and out.severity >= 1 and bool(out.summary.strip())
        usage = res.usage()
        return {
            "pass": ok,
            "latency_ms": round(ms, 1),
            "output_tokens": usage.output_tokens,
            "thinking_part": has_thinking_part(res),
            "output": out.model_dump(),
            "error": None if ok else f"Unexpected values: {out.model_dump()}",
        }
    except Exception as exc:
        return {
            "pass": False,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "output_tokens": None,
            "thinking_part": None,
            "output": None,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def run_multistep_tool_test(model_name: str) -> dict:
    """Test multi-step tool invocation + RefundVerdict decision via pydantic-ai."""
    calls: list[str] = []
    t0 = time.perf_counter()
    try:
        agent = Agent(
            make_model(model_name),
            output_type=NativeOutput(RefundVerdict),
            retries=1,
            instructions=(
                "Policy: Refunds are approved only if delivered within 30 days AND unopened/sealed. "
                "Always look up the order first using lookup_order before deciding."
            ),
        )

        @agent.tool_plain
        def lookup_order(order_id: str) -> dict:
            """Look up order delivery age, seal status, and amount."""
            calls.append(order_id)
            return {
                "order_id": order_id,
                "days_since_delivery": 12,
                "sealed": True,
                "amount_usd": 149.0,
            }

        res = agent.run_sync(
            "Customer wants a refund for order ORD-42. Should we approve_refund, deny_refund, or escalate?",
            model_settings={"temperature": 0.0, "max_tokens": 2048},
        )
        ms = (time.perf_counter() - t0) * 1000.0
        out: RefundVerdict = res.output
        tool_ok = "ORD-42" in calls
        verdict_ok = out.verdict == "approve_refund" and "ORD-42" in out.order_id
        ok = tool_ok and verdict_ok
        usage = res.usage()
        return {
            "pass": ok,
            "tool_called": tool_ok,
            "calls": calls,
            "latency_ms": round(ms, 1),
            "output_tokens": usage.output_tokens,
            "thinking_part": has_thinking_part(res),
            "output": out.model_dump(),
            "error": None if ok else f"tool_called={tool_ok}, output={out.model_dump()}",
        }
    except Exception as exc:
        return {
            "pass": False,
            "tool_called": bool(calls),
            "calls": calls,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            "output_tokens": None,
            "thinking_part": None,
            "output": None,
            "error": f"{type(exc).__name__}: {str(exc)[:240]}",
        }


def run_all() -> dict:
    """Run all tests serially across the 7 generative models."""
    results = []
    for model in MODELS:
        print(f"\n=== Benchmarking {model} ===")
        unload_all_models()

        probe = probe_speed_and_think(model)
        print(
            f"  [Probe think=false] RAM={probe['resident']['size_gb']} GB | "
            f"decode={probe['decode_tok_s']} tok/s | prefill={probe['prefill_tok_s']} tok/s | "
            f"think_false_respected={probe['think_false_respected']}"
        )

        native = run_native_output_test(model)
        print(f"  [NativeOutput] pass={native['pass']} | {native['latency_ms']} ms | out_tok={native['output_tokens']} | err={native['error']}")

        tool_out = run_tool_output_test(model)
        print(f"  [ToolOutput]   pass={tool_out['pass']} | {tool_out['latency_ms']} ms | out_tok={tool_out['output_tokens']} | err={tool_out['error']}")

        multi = run_multistep_tool_test(model)
        print(f"  [MultiStep]    pass={multi['pass']} (tool={multi['tool_called']}) | {multi['latency_ms']} ms | out_tok={multi['output_tokens']} | err={multi['error']}")

        unload_all_models()
        results.append(
            {
                "model": model,
                "probe": probe,
                "native_output": native,
                "tool_output": tool_out,
                "multistep_tool": multi,
            }
        )
    return {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"), "models": results}


def write_reports(data: dict, out_dir: Path) -> None:
    """Write JSON and Markdown benchmark reports."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gen_results_2026-11-26.json"
    md_path = out_dir / "gen_results_2026-11-26.md"
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    lines = [
        "# Local Mac Mini M4 (24GB) Generative `<10B` + `pydantic-ai` Benchmark (`2026-11-26`)",
        "",
        "Serial evaluation via Ollama `/api/chat` (`think: false` probe) and `pydantic-ai` `OllamaModel` (`/v1`). Each model was unloaded (`keep_alive: 0`) before loading the next.",
        "",
        "## Summary",
        "",
        "| Model | RAM (`/api/ps`) | Decode (tok/s) | Prefill (tok/s) | `think:false` Respected | `NativeOutput` | `ToolOutput` | Multi-Step Tool + Decision |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- |",
    ]
    for r in data["models"]:
        p = r["probe"]
        ram = f"{p['resident']['size_gb']:.2f} GB" if p["resident"]["size_gb"] is not None else "N/A"
        dec = f"{p['decode_tok_s']}" if p["decode_tok_s"] is not None else "N/A"
        pre = f"{p['prefill_tok_s']}" if p["prefill_tok_s"] is not None else "N/A"
        tf = "Yes" if p["think_false_respected"] else f"No ({p['thinking_chars']} chars)"

        def fmt(rec: dict) -> str:
            if rec["pass"]:
                return f"PASS ({rec['latency_ms']:.0f} ms, {rec['output_tokens']} tok)"
            err = (rec.get("error") or "failed").replace("|", "/").replace("\n", " ")[:60]
            return f"FAIL ({rec['latency_ms']:.0f} ms: {err})"

        lines.append(
            f"| `{r['model']}` | {ram} | {dec} | {pre} | {tf} | {fmt(r['native_output'])} | {fmt(r['tool_output'])} | {fmt(r['multistep_tool'])} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    report = run_all()
    write_reports(report, Path("/Users/mfranz/tmp/ollama-dms/benchmarks"))
