# Load test results

Setup (2026-10-22): full Docker Compose stack on one Windows laptop (Docker Desktop, 16 vCPU /
8 cores, 12 GB for Docker). Gateway with semantic cache (all-MiniLM-L6-v2); local model
**qwen2.5:1.5b-instruct base** (the fine-tuned model isn't trained yet), Ollama on CPU with 8
threads; **no Claude fallback configured**, so low-confidence answers are returned degraded and
cached. Locust: 8 users (one API key each), 0.5–2 s think time, 5 minutes, cold cache at start.
Traffic mix: 40% exact repeats of a hot set, 40% hot templates with new IPs/users, 20% cold lines.

| | Run 1: retry local timeouts | Run 2: don't retry local timeouts |
|---|---|---|
| Requests | 858 | 861 |
| Throughput | 2.86 req/s | 2.87 req/s |
| Errors | 4 (0.47%, HTTP 503) | 9 (1.05%, HTTP 503) |
| Cache hit rate (exact / semantic / miss) | 89.7% (431 / 339 / 88) | 89.1% (431 / 336 / 94) |
| p50 latency, all requests | 12 ms | 13 ms |
| p50 exact hit / semantic hit | 7 ms / 74 ms | 7 ms / 72 ms |
| p95 / p99 / max | 13 s / 21 s / 40 s | 15 s / 20 s / 20 s |
| Request-log rows written / dropped | 858 / 0 | 861 / 0 |

Raw Locust stats: `run1_retry_on_timeout_stats.csv`, `run2_no_timeout_retry_stats.csv`.

## What this shows

- **The cache carries the load.** With repetitive traffic ~89% of requests were answered from
  Redis in milliseconds. A semantic hit costs ~70 ms more than an exact hit (embedding the line on
  CPU).
- **Misses queue behind one CPU model.** Ollama answers one request at a time here (~3–6 s each),
  so about 90 misses in 5 minutes is close to its capacity. The latency tail and every error come
  from that queue.
- **Retrying a timeout on an overloaded local model makes things worse.** In run 1 a timed-out call
  was retried and re-queued, pushing the worst case to 40 s. Run 2 retries only connection errors
  for self-hosted models. The worst case is now bounded by the 20 s timeout, but requests that a
  retry would have rescued fail instead. With a fallback configured, those would go to Claude rather
  than return 503.
- **Fixing the root cause** needs capacity (more Ollama replicas, a GPU with vLLM batching) or load
  shedding: when the local queue is deep, go straight to the fallback. See DESIGN.md §10.

These numbers describe this laptop and this traffic mix only; they are not a capacity claim for any
other hardware.
