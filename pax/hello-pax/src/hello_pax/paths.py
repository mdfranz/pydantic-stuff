import os
from collections.abc import Mapping
from pathlib import Path

STATE_DIR_ENV = 'HELLO_PAX_STATE_DIR'


def state_dir(env: Mapping[str, str] | None = None) -> Path:
    """Resolve the state directory without creating it."""
    env = os.environ if env is None else env
    explicit = env.get(STATE_DIR_ENV)
    if explicit and Path(explicit).is_absolute():
        return Path(explicit)
    xdg = env.get('XDG_STATE_HOME')
    if xdg and Path(xdg).is_absolute():
        return Path(xdg) / 'hello-pax'
    return Path.home() / '.local' / 'state' / 'hello-pax'


def mkdir_private(directory: Path) -> None:
    """Create missing directories with mode 0700, leaving existing ones untouched."""
    missing: list[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for path in reversed(missing):
        path.mkdir(mode=0o700, exist_ok=True)
        path.chmod(0o700)
