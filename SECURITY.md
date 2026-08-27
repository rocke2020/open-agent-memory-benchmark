# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately to the repository owner
before opening a public issue. Include the affected revision, reproduction
steps, expected impact, and whether any credentials or persistent provider
state may have been exposed. Do not include live secrets or private benchmark
evidence in the report.

## Supported scope

Security fixes target the current development branch and the latest published
release. Historical releases may receive a documented workaround instead of a
backport.

## Local data safety

OAMB uses create-only scopes and preserves provider state. Do not test a report
against an existing database, account, collection, bank, or peer unless the
owner explicitly approved that exact target. Never use destructive cleanup to
recover from a failed run.

Credentials belong only in ignored local configuration. Logs, validation
issues, capsules, reports, commits, and issue trackers must contain variable
names or redacted fingerprints, never resolved values.
