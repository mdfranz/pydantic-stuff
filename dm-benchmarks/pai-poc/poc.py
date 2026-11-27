"""Smoke test: structured triage decision via pydantic-ai -> Ollama (OpenAI-compatible /v1)."""

import sys
from typing import Literal

import logfire
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

OLLAMA_BASE_URL = "http://localhost:11434/v1"

logfire.configure(service_name="pai-poc")
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)


class Triage(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    priority: Literal["low", "medium", "high"]
    rationale: str = Field(description="One short sentence justifying the decision.")


def make_agent(model_name: str) -> Agent[None, Triage]:
    model = OpenAIChatModel(model_name, provider=OllamaProvider(base_url=OLLAMA_BASE_URL))
    return Agent(
        model,
        output_type=Triage,
        system_prompt="You triage support tickets. Respond only with the requested structure.",
        model_settings=ModelSettings(temperature=0.0),
    )


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "granite4.2:8b-q4_K_M"
    ticket = "I was charged twice for my subscription this month and need a refund today."
    result = make_agent(model_name).run_sync(ticket)
    print(model_name, "->", result.output.model_dump_json())
