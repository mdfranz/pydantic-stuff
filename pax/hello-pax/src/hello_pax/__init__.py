"""hello-pax: a minimal isolated pydantic-clai2 shell with Logfire telemetry."""

from importlib.metadata import PackageNotFoundError, version


def package_version() -> str:
    try:
        return version('hello-pax')
    except PackageNotFoundError:
        return '0.1.0'
