# Third-Party Components

This inventory covers direct dependencies introduced by the package foundation.
Transitive dependency details and exact versions are frozen in `uv.lock`.

| Component | Role | License | Distribution |
|---|---|---|---|
| Pydantic | Strict immutable contracts and JSON Schema generation | MIT | Runtime dependency |
| Typer | CLI composition and schema commands | MIT | Runtime dependency |
| Hatchling | Python wheel and source build backend | MIT | Build dependency |
| pytest | Unit and contract test runner | MIT | Development dependency |
| Hypothesis | Canonical identity and state property tests | MPL-2.0 | Development dependency |
| Ruff | Formatting and static linting | MIT | Development dependency |
| mypy | Static type checking | MIT | Development dependency |

Provider images, source archives, datasets, PromptPacks, and imported evidence
retain their own upstream licenses and notices. Their inclusion in a local
workflow does not relicense them under Apache-2.0.
