import asyncio

import fakeredis

from loglens.gateway.cache import HashingEmbedder, TriageCache

VALUE = {"label": {"severity": "info"}, "provider": "local", "model_version": "m"}


def test_changing_embedder_does_not_mix_vector_sizes():
    async def scenario():
        r = fakeredis.FakeAsyncRedis()
        await TriageCache(r, HashingEmbedder(dim=512)).put(
            "ns", "invalid user a from 1.2.3.4", VALUE
        )
        # Restart with a different embedding size: must miss cleanly, not crash.
        other = TriageCache(r, HashingEmbedder(dim=384))
        return await other.get("ns", "invalid user a from 5.6.7.8")

    assert asyncio.run(scenario()) is None


def test_namespaces_isolate_model_versions():
    async def scenario():
        r = fakeredis.FakeAsyncRedis()
        c = TriageCache(r, HashingEmbedder())
        await c.put("v1:model-a", "disk full", VALUE)
        return await c.get("v1:model-a", "disk full"), await c.get("v1:model-b", "disk full")

    same, other = asyncio.run(scenario())
    assert same is not None and same.kind == "exact"
    assert other is None


def test_semantic_threshold_rejects_different_events():
    async def scenario():
        r = fakeredis.FakeAsyncRedis()
        c = TriageCache(r, HashingEmbedder(), semantic_threshold=0.95)
        await c.put("ns", "Invalid user admin from 10.0.0.1", VALUE)
        return await c.get("ns", "Out of Memory: Killed process 2592 (httpd).")

    assert asyncio.run(scenario()) is None
