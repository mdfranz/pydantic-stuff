import os
from collections.abc import MutableMapping

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.gateway import gateway_provider  # pyright: ignore[reportUnknownVariableType]

from .errors import HelloPaxError

GATEWAY_BASE_URL_ENV = 'PYDANTIC_AI_GATEWAY_BASE_URL'


class ModelConfigError(HelloPaxError):
    def __init__(self, message: str) -> None:
        super().__init__(message, rule_code='model-config')


def is_model_string(model: str) -> bool:
    return ':' in model or model == 'test'


def normalize_gateway_base_url(route: str, env: MutableMapping[str, str] | None = None) -> None:
    """Strip a trailing `/{route}` so the gateway provider does not append the route twice."""
    env = os.environ if env is None else env
    base = env.get(GATEWAY_BASE_URL_ENV)
    if not base:
        return
    trimmed = base.rstrip('/')
    suffix = f'/{route}'
    trimmed = trimmed.removesuffix(suffix)
    env[GATEWAY_BASE_URL_ENV] = trimmed


def resolve_model(model: str, *, gateway_route: str | None = None) -> Model | str:
    model = model.strip()
    if not model:
        raise ModelConfigError('model must not be empty')
    if gateway_route is None:
        if not is_model_string(model):
            raise ModelConfigError(
                f'model {model!r} needs a provider prefix (e.g. google:, anthropic:, openai:, openrouter:) or --gateway-route'
            )
        return model
    route = gateway_route.strip()
    if not route or '/' in route:
        raise ModelConfigError(f'invalid gateway route {gateway_route!r}: expected a single non-empty segment')
    if is_model_string(model):
        raise ModelConfigError(f'model {model!r} already has a provider prefix; drop --gateway-route')
    normalize_gateway_base_url(route)
    try:
        provider = gateway_provider('openai', route=route)
    except UserError as exc:
        raise ModelConfigError(str(exc)) from exc
    return OpenAIChatModel(model, provider=provider)


def model_display_name(model: Model | str) -> str:
    return model if isinstance(model, str) else model.model_name
