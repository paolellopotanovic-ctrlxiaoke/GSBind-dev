from m4.cli import COMMAND_MODULES, _load_config, _to_cli_args


def test_yaml_arguments_preserve_training_values(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("command: train\nargs:\n  batch_size: 2\n  amp: true\n", encoding="utf-8")
    command, args = _load_config(str(path))
    assert command == "train"
    assert args == ["--batch-size", "2", "--amp"]


def test_false_flags_are_omitted() -> None:
    assert _to_cli_args({"amp": False, "epochs": 20}) == ["--epochs", "20"]


def test_public_cli_separates_prediction_and_evaluation() -> None:
    assert COMMAND_MODULES["prepare-input"] == "m4.inputs"
    assert COMMAND_MODULES["predict"] == "m4.predictor"
    assert COMMAND_MODULES["eval"].endswith("eval_s1_surface_field")
