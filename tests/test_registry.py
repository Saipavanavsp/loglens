import yaml

from loglens.llmops.registry import flatten_metrics, update_gateway_config


def test_flatten_metrics_skips_non_numeric_and_step():
    hist = [
        {"step": 10, "loss": 1.5, "learning_rate": 2e-4, "epoch": 0.5},
        {"step": 20, "eval_loss": 1.1, "note": "x", "flag": True},
    ]
    flat = flatten_metrics(hist)
    assert ("loss", 1.5, 10) in flat
    assert ("eval_loss", 1.1, 20) in flat
    assert all(name not in {"step", "note", "flag"} for name, _, _ in flat)


def test_update_gateway_config_keeps_other_keys(tmp_path):
    path = tmp_path / "gateway.yaml"
    path.write_text(yaml.safe_dump({"routing": {"confidence_threshold": 0.7}}))
    update_gateway_config(path, "canary", "loglens:v2", "2")
    cfg = yaml.safe_load(path.read_text())
    assert cfg["routing"]["confidence_threshold"] == 0.7
    assert cfg["models"]["canary"] == {"ollama_tag": "loglens:v2", "registry_version": "2"}
