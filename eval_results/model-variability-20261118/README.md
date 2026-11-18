# Same-input MiniMax answer variability: one observed example

## Overview

With the same final retrieved context, model and complete answer request, MiniMax M3.1 Flash answered one LongMemEval question correctly in the published attempt and incorrectly in a later repeat. This is an illustrative saved example selected after observing different outcomes, not a new benchmark result or an estimate of how often scores vary. The published [476/500 campaign](../lme500-20261117/README.md) remains unchanged.

## 1. Identical answer inputs

| Input | Both attempts |
| --- | --- |
| Question ID | `e66b632c` |
| Question | What was my previous personal best time for the charity 5K run? |
| Fixed reference answer | 27 minutes and 45 seconds |
| Answer and judge model | `minimax-m3.1-flash` |
| Answer context | Byte-identical; 12,376 `o200k_base.encode_ordinary` tokens |
| Answer options | Native adaptive; `reasoning_split: true`, `temperature: 0`, `top_p: 1`, `n: 1`, `stop: null`; no explicit output cap |
| Full answer request | Identical messages and options; canonical request and message hashes retained in [example.json](example.json) |

The complete context is retained in the published [question shard](../lme500-20261117/oam/questions-401-450.json.gz), under this question ID. Its SHA-256 is `a7d77652308b82797eefc916e4d4105a0ef41f10cdf64d0a34d8d45e19d5743d`. It contains the explicit earlier 27:45 record, a later 26:30 result and an approximate improvement; both answers had the same evidence available.

## 2. Different saved outcomes

| Attempt | Visible answer summary | Recorded judge outcome |
| --- | --- | --- |
| Published | Previous best **27:45**; later 26:30 is a 1:15 improvement | Correct |
| Repeated | Previous best **about 28:00**, inferred as 26:30 plus about 1:30, despite also mentioning the explicit 27:45 record | Incorrect |

[example.json](example.json) preserves both complete visible answers and judgments, request options, request-message hashes, response model names, finish reasons, usage and original receipt hashes. Both role responses in each attempt finish with `stop`. As in the published campaign, request-message expression and separate reasoning text are omitted. Question and context evidence retain the [LongMemEval MIT notice](../lme500-20261117/LICENSES/LongMemEval-MIT.txt).

## 3. Interpretation limits

Reported accuracy reflects retrieval, answer generation and model-based judging together. This case establishes an observed difference in generated answers under identical recorded answer inputs. It does not isolate the underlying cause or measure a variability frequency, population error rate or benchmark error bar. One paired small-cohort rerun is also insufficient to establish such a rate; repeated independent evaluations would be needed to characterize a distribution.

The judges received different answers, so this example does not independently measure judge variability. Scores remain the recorded outcomes of their respective runs, rather than guarantees for another run. No usable negative judgment was retried to obtain a pass, and exporting these saved receipts makes no new model, retrieval or indexing calls. Original campaign shards, manifests and aggregate scores are unchanged.
