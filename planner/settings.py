"""Release paths and runtime settings; independent of the caller's directory."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_settings(path=None):
    with resolve_path(path or "config/inference.yaml").open() as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("Runtime configuration must be a YAML mapping")
    return settings
