from loglens.data.sample import build_sample
from loglens.data.templates import mine_templates


def _rows():
    rows = []
    for src, n_keys in [("a", 30), ("b", 5), ("c", 30)]:
        for k in range(n_keys):
            for dup in range(3):
                rows.append(
                    {
                        "id": f"{src}{k}-{dup}",
                        "source": src,
                        "key": f"evt{k % 7} x{k}",
                        "template": f"{src}:evt{k % 7}",
                    }
                )
    return rows


def test_one_row_per_key_and_template_cap():
    sample = build_sample(_rows(), 1000, per_template=2)
    keys = [(r["source"], r["key"]) for r in sample]
    assert len(keys) == len(set(keys))
    # a and c: 7 templates x 2 lines each; b: 5 templates with 1 key each
    assert len(sample) == 14 + 5 + 14


def test_sources_are_balanced_until_one_runs_out():
    sample = build_sample(_rows(), 15)
    counts = {s: sum(r["source"] == s for r in sample) for s in "abc"}
    assert counts == {"a": 5, "b": 5, "c": 5}


def test_small_sample_is_prefix_of_large_sample():
    small = build_sample(_rows(), 20)
    large = build_sample(_rows(), 50)
    assert [r["id"] for r in small] == [r["id"] for r in large[:20]]


def test_alert_rows_come_first_within_template():
    rows = [
        {"id": "1", "source": "bgl", "key": "a", "template": "t", "alert": False},
        {"id": "2", "source": "bgl", "key": "b", "template": "t", "alert": True},
    ]
    assert build_sample(rows, 1)[0]["id"] == "2"


def test_drain_groups_usernames():
    keys = {
        "openssh": [
            "input_userauth_request: invalid user tianf [preauth]",
            "input_userauth_request: invalid user xbmc [preauth]",
            "accepted password for root from <ip> port <num> ssh2",
        ]
    }
    t = mine_templates(keys)
    assert t[("openssh", keys["openssh"][0])] == t[("openssh", keys["openssh"][1])]
    assert t[("openssh", keys["openssh"][0])] != t[("openssh", keys["openssh"][2])]
