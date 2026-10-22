"""Load test for the gateway.

    pip install -e ".[loadtest]"
    locust -f infra/loadtest/locustfile.py --host http://127.0.0.1:8000 \
        --headless -u 8 -r 2 -t 5m --csv reports/loadtest/run

Set LOGLENS_API_KEYS to the gateway's comma-separated keys; users take one each.
On Windows use 127.0.0.1, not localhost (localhost tries IPv6 first and adds ~2 s
per new connection to Docker Desktop's IPv4-only port forward). Traffic mix:
  40% exact repeats of a small hot set   -> exercises the exact cache
  40% hot templates with new IPs/users   -> exercises the semantic cache
  20% lines from the cold pool           -> mostly the model
Afterwards, GET /v1/stats shows the cache hit rate for the run.
"""

import os
import random

from locust import HttpUser, between, task

TEMPLATES = [
    ("openssh", "Invalid user {user} from {ip}"),
    ("openssh", "Failed password for invalid user {user} from {ip} port {port} ssh2"),
    ("openssh", "Accepted password for {user} from {ip} port {port} ssh2"),
    ("openssh", "Received disconnect from {ip}: 11: Bye Bye [preauth]"),
    (
        "linux",
        "authentication failure; logname= uid=0 euid=0 tty=NODEVssh ruser= rhost={ip}  user={user}",
    ),
    ("linux", "session opened for user {user} by (uid=0)"),
    ("apache", "[error] [client {ip}] Directory index forbidden by rule: /var/www/html/"),
    ("apache", "[error] mod_jk child workerEnv in error state {n}"),
]
COLD = [
    ("linux", "Out of Memory: Killed process {n} (httpd)."),
    ("bgl", "KERNEL INFO instruction cache parity error corrected"),
    ("bgl", "KERNEL FATAL data TLB error interrupt"),
    ("linux", "EXT3-fs error (device sda{n}): ext3_find_entry: reading directory #2 offset 0"),
    ("linux", "eth{n}: link down"),
    ("apache", "[notice] Apache/2.0.49 configured -- resuming normal operations"),
    ("bgl", "KERNEL FATAL machine check interrupt (bit=0x{n}): L2 dcache unit data parity error"),
    ("linux", "nfs: server fileserver{n} not responding, still trying"),
]
USERS = ["admin", "oracle", "test", "guest", "postgres", "ubuntu", "git", "deploy", "pi", "ftp"]


def fill(template: str, rng: random.Random) -> str:
    return template.format(
        user=rng.choice(USERS),
        ip=f"{rng.randint(1, 223)}.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}",
        port=rng.randint(1024, 65535),
        n=rng.randint(0, 9),
    )


# A fixed hot set, identical across users, for exact-cache repeats.
_seed = random.Random(42)
HOT = [(src, fill(t, _seed)) for src, t in TEMPLATES]


class GatewayUser(HttpUser):
    wait_time = between(0.5, 2.0)

    _next_key = 0

    def on_start(self):
        # One key per simulated user (round-robin over LOGLENS_API_KEYS), like separate clients;
        # each key has its own rate limit and budget.
        keys = [k.strip() for k in os.environ["LOGLENS_API_KEYS"].split(",") if k.strip()]
        key = keys[GatewayUser._next_key % len(keys)]
        GatewayUser._next_key += 1
        self.headers = {"Authorization": f"Bearer {key}"}
        self.rng = random.Random()

    def _post(self, source: str, message: str, name: str):
        with self.client.post(
            "/v1/triage",
            json={"message": message, "source": source},
            headers=self.headers,
            name=name,
            catch_response=True,
        ) as resp:
            if resp.status_code == 429:
                resp.success()  # rate limiting is expected behaviour under load
            elif resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")

    @task(4)
    def exact_repeat(self):
        self._post(*self.rng.choice(HOT), name="triage:hot-exact")

    @task(4)
    def semantic_variant(self):
        src, t = self.rng.choice(TEMPLATES)
        self._post(src, fill(t, self.rng), name="triage:hot-variant")

    @task(2)
    def cold(self):
        src, t = self.rng.choice(COLD)
        self._post(src, fill(t, self.rng), name="triage:cold")
