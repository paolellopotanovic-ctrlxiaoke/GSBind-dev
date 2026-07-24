from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from ._bootstrap import activate_validated_implementation


COMMAND_MODULES = {
    "prepare-input": "m4.inputs",
    "predict": "m4.predictor",
    "train": "apex_s1_surface_field.train_s1_surface_field",
    "eval": "apex_s1_surface_field.eval_s1_surface_field",
    "build-tensor-cache": "processing.precompute_apex_tensor_cache",
    "build-graph-cache": "processing.precompute_apex_residue_graph_cache",
    "build-surface-cache": "apex_s1_surface_field.build_s1_surface_cache",
    "build-static-cache": "apex_s1_surface_field.build_e89_static_geometry_cache",
    "audit-manifest": "processing.audit_apex_manifest_ordinal_alignment",
}


def _to_cli_args(values: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for key, value in values.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            result.append(flag)
        elif value is False or value is None:
            continue
        elif isinstance(value, (list, tuple)):
            result.extend((flag, ",".join(str(item) for item in value)))
        else:
            result.extend((flag, str(value)))
    return result


def _load_config(path: str) -> tuple[str | None, list[str]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping.")
    command = payload.pop("command", None)
    args = payload.pop("args", payload)
    if not isinstance(args, dict):
        raise ValueError("Configuration 'args' must be a mapping.")
    return command, _to_cli_args(args)


def _dispatch(module_name: str, forwarded: list[str]) -> None:
    activate_validated_implementation()
    module = __import__(module_name, fromlist=["main"])
    original = sys.argv
    try:
        sys.argv = [module_name, *forwarded]
        module.main()
    finally:
        sys.argv = original


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    overview = argparse.ArgumentParser(prog="m4", description="Train and evaluate M4.")
    overview.add_argument("command", choices=sorted(COMMAND_MODULES))
    overview.add_argument("--config", help="YAML file containing an args mapping.")
    if not raw or raw[0] in {"-h", "--help"}:
        overview.parse_args(raw)
        return
    parser = argparse.ArgumentParser(prog="m4", add_help=False)
    parser.add_argument("command", choices=sorted(COMMAND_MODULES))
    parser.add_argument("--config", help="YAML file containing an args mapping.")
    known, forwarded = parser.parse_known_args(raw)
    command = known.command
    if known.config:
        config_command, config_args = _load_config(known.config)
        if config_command and config_command != command:
            parser.error(f"config command '{config_command}' does not match '{command}'")
        forwarded = config_args + forwarded
    _dispatch(COMMAND_MODULES[command], forwarded)


def inference_main(argv: list[str] | None = None) -> None:
    _dispatch(COMMAND_MODULES["predict"], list(sys.argv[1:] if argv is None else argv))
