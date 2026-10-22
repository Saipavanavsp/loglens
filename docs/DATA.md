# Dataset

## Sources

All data comes from [Loghub](https://github.com/logpai/loghub) (Zenodo record 8196385), which is
free for research and academic use. Raw data is downloaded by script and never committed.

| Source | Lines | What it is |
|---|---|---|
| linux | 25,567 | syslog from one Linux server |
| openssh | 655,147 | sshd on an internet-facing host, heavy brute-force traffic |
| apache | 56,482 | Apache error log |
| bgl | 4,747,963 | BlueGene/L supercomputer RAS log, with Loghub's per-line alert flag |

HDFS was dropped. Its public sample has 16 distinct messages once block ids and IPs are masked.
BGL adds kernel, memory and hardware faults, which the other three sources rarely have.

## Pipeline

```
download -> parse -> sample -> label (Claude) -> select golden -> human review -> split -> stats
```

1. **parse** (`loglens.data.parse`). A regex per source extracts timestamp, host, component and
   message. Lines that don't match are counted and skipped, mostly wrapped Apache continuation lines.
   Each message gets a masked key (IPs, numbers, hex, paths and block ids replaced), and at most 3
   lines are kept per key.
2. **templates** (`loglens.data.templates`). A compact re-implementation of the Drain log parser
   groups masked messages into event templates such as `invalid user <*> from <ip>`. The drain3
   package isn't used because its exact `cachetools` pin conflicts with MLflow.
3. **sample** (`loglens.data.sample`). One line per masked key, at most 5 per template, round-robin
   across templates and sources. The ordering is deterministic, so a small pilot sample is a prefix
   of the full sample and its labels are reused from the cache.
4. **label** (`loglens.data.label`). Claude labels batches of 10 lines against a JSON schema. Each
   label is re-validated with Pydantic, missing lines are retried individually, and results are
   cached on disk. Dry run by default, with a request cap, a cost estimate and a confirmation prompt.
5. **review** (`loglens.data.review`, `dashboard/review_app.py`). 200 golden candidates, one per
   template and balanced across sources, are reviewed by hand in a Streamlit page.
6. **split** (`loglens.data.split`):
   - golden = reviewed golden candidates only;
   - human corrections override teacher labels;
   - train/val lines that are 90% or more similar to any golden line (on masked text) are removed;
   - train/val are split by template, so validation measures generalisation to unseen event types;
   - each golden row records whether its template also appears in train (`seen_template`).
7. **stats** (`loglens.data.stats`) writes `reports/data_stats.md`.

## Label schema

```json
{"severity": "info|warning|error|critical",
 "category": "auth|network|disk|memory|service|security|other",
 "root_cause": "one or two sentences",
 "suggested_action": "one concrete step"}
```

Definitions and tie-break rules are in `prompts/teacher/v1.yaml`.

## Known limitations

- Labels come from one teacher model plus one reviewer, so expect some noise in the categories.
- Each line is judged on its own. Things that only show up across many lines, like a brute-force
  attack, can't be seen from a single line.
- Logs are from 2005-2009 systems. Modern formats (JSON logs, Kubernetes) are not covered.
