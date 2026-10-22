"""API key authentication.

Clients send `Authorization: Bearer <key>` or `X-API-Key: <key>`. Keys are
compared in constant time. Only a short hash of the key (key_id) is ever
logged or used in Redis keys, never the key itself.
"""

import hashlib
import hmac

from fastapi import HTTPException, Request


def key_id(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()[:12]


class ApiKeyAuth:
    def __init__(self, valid_keys: set[str]):
        self._hashes = {hashlib.sha256(k.encode()).digest() for k in valid_keys}

    def _extract(self, request: Request) -> str | None:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            return header[7:].strip()
        return request.headers.get("x-api-key")

    async def __call__(self, request: Request) -> str:
        presented = self._extract(request)
        if not presented:
            raise HTTPException(401, "missing API key", headers={"WWW-Authenticate": "Bearer"})
        digest = hashlib.sha256(presented.encode()).digest()
        # Compare against every key so timing doesn't reveal which one matched.
        ok = False
        for h in self._hashes:
            ok |= hmac.compare_digest(digest, h)
        if not ok:
            raise HTTPException(401, "invalid API key", headers={"WWW-Authenticate": "Bearer"})
        return key_id(presented)
