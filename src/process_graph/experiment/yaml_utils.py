from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import yaml


def _read_yaml_text(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        text = raw.decode("utf-8", errors="replace")
        print(
            "[yaml][warn] "
            f"{path} is not valid UTF-8 at byte {exc.start}; "
            "decoded with replacement characters. "
            "This is usually caused by non-UTF8 comments in a YAML config.",
            flush=True,
        )
        return text


def read_yaml_mapping(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"YAML file not found: {path}")
    loaded = yaml.safe_load(_read_yaml_text(path))
    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Expected YAML mapping at root, got {type(loaded)} in {path}")
    return dict(loaded)


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge `override` into `base` (mutates `base`)."""

    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base
