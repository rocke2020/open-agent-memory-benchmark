"""Explicit no-retry HTTP calls for the Hindsight REST profile."""

from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import quote

import httpx

from oamb.contracts.ports import ArtifactStorePort
from oamb.memory_systems.rest import (
    DEFAULT_MEMORY_SYSTEM_READ_TIMEOUT_SECONDS,
    DEFAULT_MEMORY_SYSTEM_TOTAL_TIMEOUT_SECONDS,
    SealedRestClient,
    SealedRestResponse,
)


class HindsightClient:
    def __init__(
        self,
        *,
        store: ArtifactStorePort,
        base_url: str,
        authorization: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        read_timeout_seconds: float = DEFAULT_MEMORY_SYSTEM_READ_TIMEOUT_SECONDS,
        total_timeout_seconds: float = DEFAULT_MEMORY_SYSTEM_TOTAL_TIMEOUT_SECONDS,
    ) -> None:
        headers: Mapping[str, str] = (
            {"Authorization": authorization} if authorization is not None else {}
        )
        self._rest = SealedRestClient(
            store=store,
            base_url=base_url,
            headers=headers,
            transport=transport,
            read_timeout_seconds=read_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
        )

    async def get_version(self) -> SealedRestResponse:
        return await self._rest.request("GET", "/version")

    async def list_banks(self, *, limit: int, offset: int) -> SealedRestResponse:
        return await self._rest.request(
            "GET",
            "/v1/default/banks",
            params={"limit": limit, "offset": offset},
        )

    async def create_bank(self, bank_id: str) -> SealedRestResponse:
        return await self._rest.request(
            "PUT",
            f"/v1/default/banks/{bank_id}",
            json_payload={"enable_observations": False},
            write_intent=True,
        )

    async def get_bank_config(self, bank_id: str) -> SealedRestResponse:
        return await self._rest.request("GET", f"/v1/default/banks/{bank_id}/config")

    async def get_bank_profile(self, bank_id: str) -> SealedRestResponse:
        return await self._rest.request("GET", f"/v1/default/banks/{bank_id}/profile")

    async def retain(self, bank_id: str, items: list[dict[str, object]]) -> SealedRestResponse:
        return await self._rest.request(
            "POST",
            f"/v1/default/banks/{bank_id}/memories",
            json_payload={"items": items, "async": False},
            write_intent=True,
        )

    async def list_documents(
        self,
        bank_id: str,
        *,
        limit: int,
        offset: int,
    ) -> SealedRestResponse:
        return await self._rest.request(
            "GET",
            f"/v1/default/banks/{bank_id}/documents",
            params={"limit": limit, "offset": offset},
        )

    async def get_document(self, bank_id: str, document_id: str) -> SealedRestResponse:
        return await self._rest.request(
            "GET",
            f"/v1/default/banks/{bank_id}/documents/{quote(document_id, safe='')}",
        )

    async def list_memories(
        self,
        bank_id: str,
        *,
        limit: int,
        offset: int,
        fact_type: str | None = None,
    ) -> SealedRestResponse:
        params: dict[str, str | int] = {"limit": limit, "offset": offset}
        if fact_type is not None:
            params["type"] = fact_type
        return await self._rest.request(
            "GET",
            f"/v1/default/banks/{bank_id}/memories/list",
            params=params,
        )

    async def get_memory(self, bank_id: str, memory_id: str) -> SealedRestResponse:
        return await self._rest.request(
            "GET",
            f"/v1/default/banks/{bank_id}/memories/{quote(memory_id, safe='')}",
        )

    async def list_mental_models(
        self,
        bank_id: str,
        *,
        limit: int,
        offset: int,
    ) -> SealedRestResponse:
        return await self._rest.request(
            "GET",
            f"/v1/default/banks/{bank_id}/mental-models",
            params={"detail": "full", "limit": limit, "offset": offset},
        )

    async def recall(self, bank_id: str, payload: dict[str, object]) -> SealedRestResponse:
        return await self._rest.request(
            "POST",
            f"/v1/default/banks/{bank_id}/memories/recall",
            json_payload=payload,
            request_evidence=True,
        )

    async def close(self) -> None:
        await self._rest.close()
