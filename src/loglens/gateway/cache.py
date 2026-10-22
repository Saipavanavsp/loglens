"""Two-level response cache in Redis: exact match first, then semantic.

Exact:    key = sha256(raw log line), namespaced by prompt + model version.
Semantic: the masked line (IPs, numbers, ids replaced) is embedded; a cached
          answer is reused when cosine similarity >= threshold (default 0.95).

Staleness: entries expire (TTL), and the namespace includes the prompt and
model version, so promoting a new model or prompt starts with a cold cache
instead of serving answers from the old one. Semantic hits have a shorter TTL
because they are approximate.

Semantic lookup is a brute-force cosine over the most recent N entries, which
is fine for a few thousand entries. At scale this becomes a Redis vector index
(HNSW via the Redis query engine) or a dedicated vector store; see DESIGN.md.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from loglens.data.normalize import mask


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Character 3-gram feature hashing, L2-normalised. No model download.

    Used in tests and as a fallback when sentence-transformers isn't installed.
    It captures surface similarity only.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        padded = f"  {text.lower()}  "
        for i in range(len(padded) - 2):
            h = int.from_bytes(
                hashlib.blake2b(padded[i : i + 3].encode(), digest_size=4).digest(), "little"
            )
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec


class SentenceTransformerEmbedder:
    """all-MiniLM-L6-v2: 384-dim, ~90 MB, fast enough on CPU for single lines."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device="cpu")
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True).astype(np.float32)


def make_embedder(prefer: str = "sentence-transformers") -> Embedder:
    if prefer == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder()
        except ImportError:
            pass
    return HashingEmbedder()


@dataclass
class CacheHit:
    value: dict
    kind: str  # "exact" | "semantic"
    similarity: float = 1.0


class TriageCache:
    def __init__(
        self,
        redis,
        embedder: Embedder | None,
        *,
        exact_ttl_s: int = 86400,
        semantic_ttl_s: int = 21600,
        semantic_threshold: float = 0.95,
        semantic_max_entries: int = 5000,
        prefix: str = "lc",
    ):
        self.r = redis
        self.embedder = embedder
        self.exact_ttl_s = exact_ttl_s
        self.semantic_ttl_s = semantic_ttl_s
        self.threshold = semantic_threshold
        self.max_entries = semantic_max_entries
        self.prefix = prefix

    @staticmethod
    def _h(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _ns(self, namespace: str) -> str:
        return f"{self.prefix}:{namespace}"

    def _sem(self, ns: str) -> str:
        # The embedding size is part of the key: vectors from a different embedder
        # (e.g. after a fallback to HashingEmbedder) must never be compared.
        return f"{ns}:sem{self.embedder.dim}"

    async def _embed(self, text: str) -> np.ndarray:
        # Model inference is CPU-bound; keep it off the event loop.
        return await asyncio.to_thread(self.embedder.embed, text)

    async def get(self, namespace: str, message: str) -> CacheHit | None:
        ns = self._ns(namespace)
        raw = await self.r.get(f"{ns}:exact:{self._h(message)}")
        if raw is not None:
            await self.r.incr(f"{self.prefix}:stats:exact_hit")
            return CacheHit(json.loads(raw), "exact")
        if self.embedder is not None:
            hit = await self._semantic_get(ns, message)
            if hit is not None:
                await self.r.incr(f"{self.prefix}:stats:semantic_hit")
                return hit
        await self.r.incr(f"{self.prefix}:stats:miss")
        return None

    async def _semantic_get(self, ns: str, message: str) -> CacheHit | None:
        sem = self._sem(ns)
        ids = await self.r.zrevrange(f"{sem}:ids", 0, self.max_entries - 1)
        if not ids:
            return None
        ids = [i.decode() if isinstance(i, bytes) else i for i in ids]
        vecs = await self.r.mget([f"{sem}:vec:{i}" for i in ids])
        live = [(i, v) for i, v in zip(ids, vecs, strict=True) if v is not None]
        expired = [i for i, v in zip(ids, vecs, strict=True) if v is None]
        if expired:
            await self.r.zrem(f"{sem}:ids", *expired)
        if not live:
            return None
        query = await self._embed(mask(message))
        matrix = np.stack([np.frombuffer(v, dtype=np.float32) for _, v in live])
        sims = matrix @ query
        best = int(np.argmax(sims))
        if float(sims[best]) < self.threshold:
            return None
        raw = await self.r.get(f"{sem}:val:{live[best][0]}")
        if raw is None:
            return None
        return CacheHit(json.loads(raw), "semantic", float(sims[best]))

    async def put(
        self, namespace: str, message: str, value: dict, semantic_value: dict | None = None
    ) -> None:
        """Store an answer.

        `semantic_value` is what a *different but similar* line will get back. The
        caller passes a version with PII scrubbed from the explanation, so one
        request's IPs or usernames are never shown to another request.
        """
        ns = self._ns(namespace)
        await self.r.set(f"{ns}:exact:{self._h(message)}", json.dumps(value), ex=self.exact_ttl_s)
        if self.embedder is None:
            return
        masked = mask(message)
        entry_id = self._h(masked)[:20]
        vec = (await self._embed(masked)).astype(np.float32).tobytes()
        sem = self._sem(ns)
        pipe = self.r.pipeline()
        pipe.set(f"{sem}:vec:{entry_id}", vec, ex=self.semantic_ttl_s)
        pipe.set(
            f"{sem}:val:{entry_id}", json.dumps(semantic_value or value), ex=self.semantic_ttl_s
        )
        pipe.zadd(f"{sem}:ids", {entry_id: time.time()})
        # Keep only the newest entries in the index.
        pipe.zremrangebyrank(f"{sem}:ids", 0, -self.max_entries - 1)
        await pipe.execute()

    async def stats(self) -> dict[str, float]:
        keys = ("exact_hit", "semantic_hit", "miss")
        vals = await self.r.mget([f"{self.prefix}:stats:{k}" for k in keys])
        counts = {k: int(v or 0) for k, v in zip(keys, vals, strict=True)}
        total = sum(counts.values())
        counts["hit_rate"] = (
            (counts["exact_hit"] + counts["semantic_hit"]) / total if total else 0.0
        )
        return counts
