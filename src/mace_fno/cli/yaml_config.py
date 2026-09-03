"""YAML-backed argument parsing shared by command-line entry points."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the common optional YAML configuration argument."""
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "YAML configuration file. Keys may be flat or grouped into named "
            "sections; explicit command-line options override YAML values."
        ),
    )


def _flatten_mapping(
    mapping: Mapping[Any, Any],
    *,
    prefix: tuple[str, ...] = (),
) -> list[tuple[str, Any, str]]:
    """Return ``(normalized leaf key, value, YAML location)`` triples."""
    flattened: list[tuple[str, Any, str]] = []
    for raw_key, value in mapping.items():
        if not isinstance(raw_key, str):
            location = ".".join(prefix) or "<root>"
            raise ValueError(f"YAML keys below {location} must be strings")
        location_parts = (*prefix, raw_key)
        location = ".".join(location_parts)
        if isinstance(value, Mapping):
            flattened.extend(_flatten_mapping(value, prefix=location_parts))
            continue
        normalized = raw_key.strip().replace("-", "_")
        if not normalized:
            raise ValueError(f"empty YAML option name at {location}")
        flattened.append((normalized, value, location))
    return flattened


def _preferred_option(action: argparse.Action, *, negative: bool = False) -> str:
    long_options = [
        option for option in action.option_strings if option.startswith("--")
    ]
    if negative:
        candidates = [option for option in long_options if option.startswith("--no-")]
    else:
        candidates = [
            option for option in long_options if not option.startswith("--no-")
        ]
    if not candidates:
        direction = "negative " if negative else ""
        raise ValueError(f"option {action.dest!r} has no {direction}long form")
    return candidates[0]


def _path_value(value: Any, config_directory: Path) -> str:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = config_directory / path
    return str(path.resolve())


def _yaml_value_tokens(
    action: argparse.Action,
    value: Any,
    *,
    config_directory: Path,
    location: str,
) -> list[str]:
    """Translate one YAML value into tokens validated later by argparse."""
    if isinstance(action, argparse.BooleanOptionalAction):
        if not isinstance(value, bool):
            raise ValueError(f"{location} must be true or false")
        return [_preferred_option(action, negative=not value)]

    option = _preferred_option(action)
    expects_sequence = action.nargs in {"+", "*"} or isinstance(action.nargs, int)
    if expects_sequence:
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(f"{location} must be a YAML sequence")
        values = list(value)
    else:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            raise ValueError(f"{location} must be a scalar value")
        values = [value]

    tokens = [option]
    for item in values:
        if action.type is Path:
            tokens.append(_path_value(item, config_directory))
        else:
            tokens.append(str(item))
    return tokens


def _load_yaml_tokens(
    parser: argparse.ArgumentParser,
    config_path: Path,
) -> list[str]:
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"cannot read YAML configuration {config_path}: {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ValueError(
            f"invalid YAML configuration {config_path}: {error}"
        ) from error

    if document is None:
        document = {}
    if not isinstance(document, Mapping):
        raise ValueError("the YAML configuration root must be a mapping")

    actions = {
        action.dest: action
        for action in parser._actions
        if action.option_strings and action.dest not in {"help", "config"}
    }
    seen: dict[str, str] = {}
    tokens: list[str] = []
    for key, value, location in _flatten_mapping(document):
        if key in seen:
            raise ValueError(
                f"duplicate YAML option {key!r} at {seen[key]} and {location}"
            )
        seen[key] = location
        action = actions.get(key)
        if action is None:
            raise ValueError(f"unknown YAML option {location!r}")
        if value is None:
            continue
        tokens.extend(
            _yaml_value_tokens(
                action,
                value,
                config_directory=config_path.parent,
                location=location,
            )
        )
    return tokens


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments after prepending values loaded from ``--config``."""
    command_line = list(sys.argv[1:] if argv is None else argv)
    if "-h" in command_line or "--help" in command_line:
        return parser.parse_args(command_line)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    preliminary, _ = config_parser.parse_known_args(command_line)
    if preliminary.config is None:
        return parser.parse_args(command_line)

    config_path = preliminary.config.expanduser().resolve()
    try:
        yaml_tokens = _load_yaml_tokens(parser, config_path)
    except ValueError as error:
        parser.error(str(error))
    namespace = parser.parse_args([*yaml_tokens, *command_line])
    namespace.config = config_path
    return namespace


def resolved_configuration(
    namespace: argparse.Namespace,
    **effective_values: Any,
) -> dict[str, Any]:
    """Return a YAML-serializable record of the effective training options."""
    values = {**vars(namespace), **effective_values}
    resolved: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Path):
            resolved[key] = str(value.expanduser().resolve())
        elif isinstance(value, tuple):
            resolved[key] = list(value)
        else:
            resolved[key] = value
    return resolved


def write_resolved_configuration(
    path: str | Path,
    configuration: Mapping[str, Any],
) -> Path:
    """Write the effective configuration next to a training checkpoint."""
    output = Path(path)
    output.write_text(
        yaml.safe_dump(dict(configuration), sort_keys=False),
        encoding="utf-8",
    )
    return output
