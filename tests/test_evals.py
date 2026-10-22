import json

import pytest

from loglens.data.io import write_jsonl
from loglens.evals import run as run_mod
from loglens.evals.errors import analyse, failures
from loglens.evals.judge import agreement
from loglens.evals.metrics import compute, compute_by_group, confusion, percentile
from loglens.evals.regression import check, rescore
from loglens.evals.report import comparison_table, fmt
from loglens.evals.systems import System
from loglens.gateway.providers.base import Completion


def lab(sev="error", cat="disk"):
    return {
        "severity": sev,
        "category": cat,
        "root_cause": "disk is full",
        "suggested_action": "free space",
    }


def pred(gold, p, latency=100.0, cost=0.0, **kw):
    return {
        "id": kw.pop("id", "x"),
        "gold": gold,
        "pred": p,
        "latency_ms": latency,
        "cost_usd": cost,
        **kw,
    }


def test_invalid_answers_count_as_wrong():
    preds = [pred(lab(), lab()), pred(lab(), None), pred(lab("info", "auth"), lab("info", "auth"))]
    m = compute(preds)
    assert m["valid_json_rate"] == pytest.approx(2 / 3)
    assert m["severity_acc"] == pytest.approx(2 / 3)
    assert m["category_acc"] == pytest.approx(2 / 3)


def test_macro_f1_is_over_gold_classes():
    preds = [pred(lab("error"), lab("error")), pred(lab("info"), lab("error"))]
    # error: P=0.5 R=1 F1=0.667; info: F1=0 -> macro 0.333
    assert compute(preds)["severity_macro_f1"] == pytest.approx(1 / 3, abs=1e-3)


def test_latency_percentiles_and_cost():
    preds = [pred(lab(), lab(), latency=v, cost=0.001) for v in range(1, 101)]
    m = compute(preds)
    assert m["latency_p50_ms"] == pytest.approx(50.5, abs=1)
    assert m["latency_p95_ms"] == pytest.approx(95, abs=1.5)
    assert m["cost_per_1k_usd"] == pytest.approx(1.0)
    assert percentile([], 50) is None


def test_group_and_confusion():
    preds = [
        pred(lab(), lab(), seen_template=True),
        pred(lab(), None, seen_template=False),
    ]
    groups = compute_by_group(preds, "seen_template")
    assert groups["True"]["severity_acc"] == 1.0 and groups["False"]["severity_acc"] == 0.0
    cols, cm = confusion(preds, "severity")
    assert cols[-1] == "invalid"
    assert cm[cols.index("error")][cols.index("invalid")] == 1


def test_judge_agreement():
    a = agreement([(5, 5), (4, 5), (2, 4), (1, 1)])
    assert a["exact_agreement"] == 0.5
    assert a["within_1_agreement"] == 0.75
    assert -1 <= a["weighted_kappa"] <= 1


def test_regression_check_and_rescore():
    rows = [
        {"gold": lab(), "raw": json.dumps(lab()), "latency_ms": 1},
        {"gold": lab(), "raw": "oops", "latency_ms": 1},
    ]
    m = compute(rescore(rows))
    assert m["valid_json_rate"] == 0.5
    assert check(m, {"valid_json_rate": 0.9}) == ["valid_json_rate = 0.5 < 0.9"]
    assert check(m, {"valid_json_rate": 0.5}) == []


def test_error_analysis_groups_by_category():
    preds = [
        pred(lab("error", "disk"), lab("warning", "disk"), message="disk full"),
        pred(lab("info", "auth"), None, message="login", parse_error="no_json_object"),
        pred(lab("info", "auth"), lab("info", "auth"), message="login ok"),
    ]
    assert len(failures(preds)) == 2
    md = analyse(preds)
    assert "### auth: 1/2 wrong" in md and "| severity | error | warning | 1 |" in md


def test_report_formatting():
    assert fmt(None, "pct") == "n/a" and fmt(0.5, "pct") == "50.0%" and fmt(1500, "ms") == "1.50 s"
    table = comparison_table({"base": {"n": 2, "severity_acc": 1.0}})
    assert "| base | 2 | 100.0% |" in table


class ScriptedProvider:
    name = "fake"
    model = "fake-model"
    external = False

    async def complete(self, messages, **kw):
        line = messages[-1]["content"]
        if "garbage" in line:
            return Completion(text="I think it is bad", model=self.model)
        return Completion(text=json.dumps(lab()), model=self.model)


def test_end_to_end_run_writes_report(tmp_path, monkeypatch):
    golden = tmp_path / "golden.jsonl"
    write_jsonl(
        golden,
        [
            {
                "id": "1",
                "source": "linux",
                "message": "disk full on /var",
                "label": lab(),
                "seen_template": True,
            },
            {
                "id": "2",
                "source": "linux",
                "message": "garbage",
                "label": lab(),
                "seen_template": False,
            },
        ],
    )
    monkeypatch.setattr(
        run_mod,
        "build_systems",
        lambda names, s, tag: [System(n, ScriptedProvider()) for n in names],
    )
    code = run_mod.main(
        [
            "--golden",
            str(golden),
            "--systems",
            "base,finetuned",
            "--out",
            str(tmp_path / "out"),
            "--no-db",
            "--no-mlflow",
        ]
    )
    assert code == 0
    run_dir = next(p for p in (tmp_path / "out").iterdir() if p.is_dir())
    metrics = json.loads((run_dir / "metrics.json").read_text())["results"]
    assert metrics["base"]["valid_json_rate"] == 0.5
    assert (run_dir / "cm_finetuned_category.png").exists()
    assert "| finetuned | 2 | 50.0%" in (tmp_path / "out" / "latest.md").read_text()
