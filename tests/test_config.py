from loglens.config import Settings


def test_api_key_is_not_leaked_in_repr(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-very-secret")
    s = Settings(_env_file=None)
    assert "very-secret" not in repr(s)
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-very-secret"


def test_api_key_set_parses_comma_list(monkeypatch):
    monkeypatch.setenv("LOGLENS_API_KEYS", "a, b,,c ")
    assert Settings(_env_file=None).api_key_set == {"a", "b", "c"}


def test_dry_run_defaults_to_true(monkeypatch):
    monkeypatch.delenv("LOGLENS_DRY_RUN", raising=False)
    assert Settings(_env_file=None).dry_run is True
