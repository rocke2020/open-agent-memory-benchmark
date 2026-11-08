# Investigations

## Overview

This directory contains focused investigations of unexpected runtime behavior, performance differences, and operational observations. These reports explain a specific phenomenon using dated evidence; they are separate from product design documents and final benchmark results.

Use descriptive filenames beginning with the investigation date: `YYYY-MM-DD-<topic>.md`. Distinguish observations, confirmed mechanisms, plausible explanations, and untested changes. Keep credentials, raw benchmark content, and machine-specific private paths out of these reports.

## Reports

[OpenViking and Mem0 ingestion latency under shared local embeddings](2026-09-11-openviking-mem0-ingestion-and-mac-load.md) examines repeated OpenViking commit delays, embedding request patterns, and the reduction in Mac load as Mem0 finished.

[Hindsight implementation comparison](hindsight-implementation-comparison.md) compares the pinned OAMB LME-60 REST path with the reference AMB implementation, including recall levels, token limits and evidence boundaries.

[Mem0 evaluation and temporal inputs](mem0-longmemeval-evaluation-and-temporal-parity.md) compares the saved LME-60 OSS result with managed Mem0 evaluation and explains the historical-timestamp adaptation.

[OpenViking evaluation comparison](openviking-longmemeval-evaluation-comparison-and-design.md) compares the OAMB LME-60 path with the pinned official self-evaluation, including prompts, retrieval and result coverage.
