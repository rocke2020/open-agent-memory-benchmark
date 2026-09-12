# Retrieval top-k semantics across memory providers

## Overview

OAMB currently sets `retrieval.top_k: 150` as one shared target ceiling on native evidence candidates visible to its generic answer pipeline. This does not mean that all three providers receive the same native request parameter or perform the same amount of retrieval work. Mem0 and OpenViking receive a provider-side limit of 150 and OAMB relies on those APIs to honor it, while the pinned Hindsight API has no `top_k`, `limit`, or `max_results` field on recall; OAMB therefore normalizes Hindsight's returned facts and hydrated source chunks and keeps the first 150 candidates in that normalized order.

The precise configuration claim is: **under the current implementation, OAMB requests an OAMB-visible native evidence ceiling of 150 candidates from Mem0 and OpenViking, while Hindsight 0.9.2 is clamped to that ceiling by OAMB after native recall.** A completed run must still prove its actual candidate counts. It would be incorrect to claim that every provider is queried with `top_k=150`, that Hindsight natively supports this parameter, that the three providers perform equivalent internal search work, or that a historical result used this configuration merely because its observed candidate count happened to be at most 150.

## 1. Version and source identity

This analysis is frozen on 2026-09-12 against OAMB base commit `fb35c8a38a85b67a2af2c9e6928c72e3c7aadb4d` plus the `retrieval.top_k: 150` implementation introduced with this document. The provider identities below are the versions actually pinned by `provider-services/versions.env`; newer checkout HEADs are not benchmark identities.

| Provider | Used version | Used source commit | Used image or source identity |
| --- | --- | --- | --- |
| Hindsight | `0.9.2` | `ebad478240d3171bb88201ececda5e8d9883d22d` | `ghcr.io/vectorize-io/hindsight-api:0.9.2-slim@sha256:7635a15739361dbdf221ba796ad25a813f876144fe113022eea8e26cb6ee75e7` |
| Mem0 | `2.0.19` | `dc82354e143c2581d505d581a00286d6ef8c3605` | Source archive SHA-256 `5443d9dd99196e33fdde31ef663518c022a8f4a86ef032e7b4bd7b00285705c2` |
| OpenViking | `0.4.16` | `499995f3ed2e7f551a715179c4053772c51ff819` | `ghcr.io/volcengine/openviking:v0.4.16@sha256:46f9e34cd37238c28cbd9535033773d179006bdf7f3e528dd1c46567abce7701` |

The formal provider release cutoff recorded by OAMB is 2026-08-27. The authoritative pins are [`provider-services/versions.env`](../../provider-services/versions.env), and the Compose image identities are frozen in [`provider-services/compose.yaml`](../../provider-services/compose.yaml).

## 2. Shared OAMB configuration and propagation

[`configs/benchmark.yml`](../../configs/benchmark.yml) owns the shared value:

```yaml
retrieval:
  generation: disabled
  top_k: 150
```

The configuration parser requires a positive integer, the resolved plan freezes the value, live execution copies it into `NativeRunControl.retrieval_top_k`, and each question receives a `RetrievalRequest(top_k=150)`. The authoritative OAMB flow is:

```text
configs/benchmark.yml
  -> RetrievalConfiguration.top_k
  -> ResolvedRetrieval.top_k
  -> NativeRunControl.retrieval_top_k
  -> RetrievalRequest.top_k
  -> provider adapter
```

Relevant implementations are [`src/oamb/config/benchmark.py`](../../src/oamb/config/benchmark.py), [`src/oamb/config/doctor.py`](../../src/oamb/config/doctor.py), [`src/oamb/live.py`](../../src/oamb/live.py), and [`src/oamb/runtime/native_run.py`](../../src/oamb/runtime/native_run.py).

## 3. Provider behavior at top-k 150

| Provider | Native request | Provider-side ceiling | OAMB post-response ceiling | What counts as one candidate |
| --- | --- | --- | --- | --- |
| Mem0 | `POST /search` with `top_k: 150` | Yes | No second slice in the adapter | One returned Mem0 memory |
| OpenViking | `POST /api/v1/search/find` with `limit: 150` | Yes | No second slice in the adapter | One returned OpenViking memory hit |
| Hindsight | `recall` with no count parameter | No | `normalize_recall(... )[:150]` | One fact or one separately hydrated source chunk |

The common value therefore defines the intended maximum number of candidates handed from the memory adapter into OAMB, but the enforcement boundary and candidate unit differ. Hindsight is locally clamped; Mem0 and OpenViking require provider contract conformance because their adapters do not apply a second slice.

## 4. Mem0 semantics

OAMB encodes the configured value directly in the native request:

```json
{
  "query": "How many fun runs did I miss because of work?",
  "filters": {"run_id": "<isolated-run-id>"},
  "top_k": 150,
  "threshold": 0.1
}
```

The adapter path is [`src/oamb/memory_systems/mem0/adapter.py`](../../src/oamb/memory_systems/mem0/adapter.py), and the exact JSON encoder is [`src/oamb/memory_systems/mem0/wire.py`](../../src/oamb/memory_systems/mem0/wire.py). Native reranking is disabled in the OAMB profile, and the adapter preserves the returned order.

At Mem0 commit `dc82354e143c2581d505d581a00286d6ef8c3605`, `mem0/memory/main.py:1379-1464` defines `search(..., top_k=20)` and assigns `limit = top_k`. Its vector-search path internally over-fetches `max(limit * 4, 60)` semantic and keyword candidates, combines scores, and calls `score_and_rank(..., top_k=limit)` at lines 1628-1687. With OAMB `top_k=150`, Mem0 may inspect up to 600 candidates per internal retrieval arm before returning at most 150 ranked memories. Provider-side `top_k` limits output count, not necessarily internal work.

If Mem0 finds only 87 memories above its threshold, the adapter receives 87. If it finds more, the provider is expected to return at most 150. OAMB validates the returned records against the sealed projection but currently relies on the Mem0 API contract rather than applying a second local slice.

## 5. OpenViking semantics

The active LongMemEval cell uses the OpenViking session REST profile and sends:

```json
{
  "query": "How many fun runs did I miss because of work?",
  "target_uri": "<isolated-memory-root>",
  "context_type": "memory",
  "limit": 150
}
```

The adapter sends `request.top_k` as `limit` in [`src/oamb/memory_systems/openviking/session_adapter.py`](../../src/oamb/memory_systems/openviking/session_adapter.py). The separate resource adapter follows the same provider-side limit rule in [`src/oamb/memory_systems/openviking/adapter.py`](../../src/oamb/memory_systems/openviking/adapter.py).

At OpenViking commit `499995f3ed2e7f551a715179c4053772c51ff819`, `openviking/server/routers/search.py:121-140` declares `FindRequest.limit`, and the `/find` handler resolves it and passes it to `service.search.find(..., limit=actual_limit)` at lines 289-319. `openviking/service/search_service.py:125-161` describes and forwards this value as the maximum result count.

If OpenViking returns 63 hits, OAMB exposes 63. If more matches exist, OpenViking is expected to return its native top 150 in provider order. As with Mem0, OAMB currently relies on the provider limit contract instead of adding another local slice.

## 6. Hindsight semantics and the top-k mismatch

OAMB sends a Hindsight recall request containing the query, `types=[world, experience]`, query timestamp, trace, and hydrated chunks. It does not send `top_k` because the pinned Hindsight API does not accept one:

```json
{
  "query": "How many fun runs did I miss because of work?",
  "types": ["world", "experience"],
  "query_timestamp": "<question-time>",
  "trace": true,
  "include": {"entities": null, "chunks": {}}
}
```

At Hindsight commit `ebad478240d3171bb88201ececda5e8d9883d22d`, `hindsight-api-slim/hindsight_api/api/http.py:284-350` defines `RecallRequest` with `budget`, `max_tokens`, types, include options, score floors, tags, and temporal controls, but no `top_k`, `limit`, or `max_results`. The newer inspected Hindsight checkout HEAD `4cc131c0b238c8f206def60804d7b6591f6a45e7` also lacks such a recall field, so this is not only an artifact of the benchmark pin.

Hindsight performs its own retrieval, fusion, optional reranking, scoring, and ordering before returning. In the pinned version, `hindsight-api-slim/hindsight_api/engine/memory_engine.py:6853` sorts scored results by final `weight` descending, and lines 6964-6966 retain at most `thinking_budget * 2` candidates for subsequent token filtering. `budget` and `max_tokens` are not substitutes for top-k: `budget` controls search breadth, while `max_tokens` controls returned fact text volume.

OAMB does not perform an additional relevance sort. [`src/oamb/memory_systems/hindsight/normalize.py`](../../src/oamb/memory_systems/hindsight/normalize.py) walks facts in Hindsight's returned order, inserts the first occurrence of each hydrated source chunk immediately after the fact that references it, appends any remaining chunks, and assigns consecutive native ranks. [`src/oamb/memory_systems/hindsight/adapter.py`](../../src/oamb/memory_systems/hindsight/adapter.py) then takes the first `request.top_k` candidates.

For example:

```text
Hindsight ordered facts: F1, F2, F3
Hydrated chunks:         F1 -> C1, F3 -> C2
OAMB normalized order:  F1, C1, F2, F3, C2
OAMB top-k operation:   normalized_candidates[:150]
```

This is a structural normalization, not a second score-based rerank. It also means Hindsight's source chunks consume top-k slots. If Hindsight returns 140 facts plus 20 separately hydrated chunks, OAMB does not expose 150 facts; it exposes the first 150 items in the interleaved fact-and-chunk sequence, and later facts or chunks are excluded depending on that sequence.

## 7. Fairness boundary and release wording

The current design aligns one specific target boundary: a conforming provider response exposes no more than 150 native evidence candidates to the common OAMB answer and judge workflow. It does not by itself prove fairness, provider contract conformance, identical provider request semantics, identical internal retrieval breadth, identical content units, or identical computation cost. Each completed comparison must validate the actual candidate counts recorded in its artifacts.

Use these claims:

- `retrieval.top_k=150` is the configured target for the common OAMB-visible native evidence ceiling.
- Mem0 2.0.19 and OpenViking 0.4.16 receive the requested ceiling as a native provider request parameter.
- Hindsight 0.9.2 has no native recall count parameter, so OAMB clamps the normalized response to 150 candidates without re-sorting it.
- Hindsight facts and separately hydrated chunks each count as candidates under the current adapter contract.

Do not use these claims:

- All three providers are queried with native `top_k=150`.
- Hindsight 0.9.2 supports the same limit parameter as Mem0 and OpenViking.
- The three providers inspect or rank exactly 150 internal records.
- The current Hindsight limit means the top 150 facts, because hydrated chunks also consume slots.
- A prior Hindsight or OpenViking result was configured with `top_k=150` solely because its recorded candidate count did not exceed 150.
- All three providers completed a new `top_k=150` comparison unless the corresponding run artifacts and actual candidate counts have been validated.

## 8. Design issue and available resolutions

The unresolved semantic issue is whether benchmark `top_k` should mean provider-native result count or OAMB-visible evidence-unit count. The current implementation chooses the second meaning because Hindsight 0.9.2 cannot accept an exact native count and because OAMB models hydrated chunks as independently attributable evidence.

Three resolutions are possible, but each changes the comparison contract:

1. Keep the current implementation and disclose the boundary precisely. This preserves the existing common visible-candidate ceiling and requires no provider-specific hidden approximation.
2. Cap Hindsight facts before chunk hydration. This would make the fact limit easier to describe, but the final visible candidate count could exceed 150 unless chunks receive a separate budget, and it would no longer match the current candidate accounting contract.
3. Require a future Hindsight release with a native exact-count recall parameter. This would align the request boundary with Mem0 and OpenViking, but it would change the pinned provider version and require a new benchmark run; provider-side internal work could still differ.

Until the comparison contract changes, the first resolution is the accurate description of artifacts produced and validated under the current configuration. Historical artifacts retain the configuration and implementation identity under which they were produced and must not be retroactively relabeled.

## 9. Source verification commands

The provider conclusions can be reproduced from checkouts containing the pinned commits:

```bash
git -C hindsight show ebad478240d3171bb88201ececda5e8d9883d22d:hindsight-api-slim/hindsight_api/api/http.py
git -C hindsight show ebad478240d3171bb88201ececda5e8d9883d22d:hindsight-api-slim/hindsight_api/engine/memory_engine.py
git -C mem0 show dc82354e143c2581d505d581a00286d6ef8c3605:mem0/memory/main.py
git -C OpenViking show 499995f3ed2e7f551a715179c4053772c51ff819:openviking/server/routers/search.py
git -C OpenViking show 499995f3ed2e7f551a715179c4053772c51ff819:openviking/service/search_service.py
```
