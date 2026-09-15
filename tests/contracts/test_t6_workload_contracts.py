from __future__ import annotations

import hashlib
from typing import Any

import pytest
from pydantic import ValidationError

from oamb.contracts.specifications import (
    MetricSpec,
    OutputContract,
    PromptPackManifest,
    PromptTemplateManifest,
    prompt_pack_manifest_hash,
)
from oamb.workloads.fake import GeneratedFakeWorkload

HASH_A = "a" * 64
HASH_B = "b" * 64


def _prompt_pack() -> PromptPackManifest:
    values: Any = dict(
        prompt_pack_id="oamb-lme-answer-v1",
        prompt_pack_version="1",
        workload_id="lme30-native-smoke-plus-v1",
        origin="oamb_authored",
        source_repository="https://example.invalid/source",
        source_revision="revision-1",
        source_file_sha256=HASH_A,
        license_expression="MIT",
        rights_attestation="project_licensed_source",
        redistribution_allowed=True,
        templates=(
            PromptTemplateManifest(
                template_name="answer_user",
                relative_path="answer-user.txt",
                content_sha256=HASH_B,
                byte_count=42,
            ),
        ),
        variables=("question", "retrieved_context"),
        output_contract_id="lme-answer-text-v1",
    )
    return PromptPackManifest(
        manifest_sha256=prompt_pack_manifest_hash(values),
        **values,
    )


def test_prompt_pack_identity_and_closed_variables_are_public_contracts() -> None:
    pack = _prompt_pack()

    assert pack.manifest_sha256 == prompt_pack_manifest_hash(
        pack.model_dump(mode="python", exclude={"manifest_sha256"})
    )

    with pytest.raises(ValidationError, match="manifest hash"):
        PromptPackManifest(**(pack.model_dump() | {"workload_id": "different"}))
    with pytest.raises(ValidationError, match="duplicate variable"):
        PromptPackManifest(
            **(
                pack.model_dump()
                | {
                    "variables": ("question", "question"),
                    "manifest_sha256": HASH_A,
                }
            )
        )
    with pytest.raises(ValidationError, match="secret"):
        PromptPackManifest(
            **(
                pack.model_dump()
                | {
                    "variables": ("question", "api_key"),
                    "manifest_sha256": HASH_A,
                }
            )
        )


def test_output_and_metric_contracts_reject_implicit_or_unknown_semantics() -> None:
    output = OutputContract(
        output_contract_id="mab-redial-ranked-movies-v1",
        representation="ranked_text_list",
        parser_id="mab-redial-ranked-movies-v1",
        max_output_tokens=512,
        required_candidate_count=1,
        accepted_finish_dispositions=("normal_stop",),
        parse_failure_policy="terminal_error",
    )
    metric = MetricSpec(
        metric_id="mab-redial-recall-at-5-v1",
        output_contract_id=output.output_contract_id,
        normalizer_id="mab-redial-title-normalizer-v1",
        scorer_id="mab-redial-recall-at-5-v1",
        answer_set_policy="preserve_occurrences",
        input_fields=("resolved_predictions", "gold_entity_ids"),
        judge_prompt_pack_id=None,
        failure_semantics="unavailable",
        unjudged_semantics="not_applicable",
    )

    assert metric.output_contract_id == output.output_contract_id
    with pytest.raises(ValidationError, match="finish disposition"):
        OutputContract(**(output.model_dump() | {"accepted_finish_dispositions": ()}))
    with pytest.raises(ValidationError, match="judge prompt"):
        MetricSpec(
            **(
                metric.model_dump()
                | {
                    "judge_prompt_pack_id": "unexpected-judge",
                    "scorer_id": "mab-redial-recall-at-5-v1",
                }
            )
        )


def test_prompt_template_content_hash_is_not_a_nonempty_proxy() -> None:
    content = b"Question: {{question}}\nContext: {{retrieved_context}}"
    template = _prompt_pack().templates[0]

    assert hashlib.sha256(content).hexdigest() != template.content_sha256
    with pytest.raises(ValidationError, match="source-extracted"):
        PromptTemplateManifest(
            template_name="bad",
            relative_path="bad.txt",
            content_sha256=HASH_A,
            byte_count=1,
            source_extracted_sha256=HASH_B,
            adaptation_id=None,
        )

    with pytest.raises(ValidationError):
        PromptTemplateManifest(
            template_name="empty",
            relative_path="empty.txt",
            content_sha256=hashlib.sha256(b"").hexdigest(),
            byte_count=0,
        )


def test_attributed_prompt_pack_requires_complete_source_provenance() -> None:
    pack = _prompt_pack()
    values = pack.model_dump(exclude={"manifest_sha256"}) | {
        "origin": "attributed_source",
        "source_repository": None,
        "source_revision": None,
        "source_file_sha256": None,
    }

    with pytest.raises(ValidationError, match="attributed"):
        PromptPackManifest(
            **values,
            manifest_sha256=prompt_pack_manifest_hash(values),
        )


def test_attributed_prompt_pack_requires_extraction_identity_for_every_template() -> None:
    pack = _prompt_pack()
    values = pack.model_dump(exclude={"manifest_sha256"}) | {
        "origin": "attributed_source",
    }

    with pytest.raises(ValidationError, match="source-extracted"):
        PromptPackManifest(
            **values,
            manifest_sha256=prompt_pack_manifest_hash(values),
        )


def test_case_manifest_hash_is_recomputed_from_its_exact_members() -> None:
    workload = GeneratedFakeWorkload()
    manifest = workload.build_case_manifest(workload.resolve_sources())

    with pytest.raises(ValidationError, match="manifest hash"):
        manifest.__class__(**(manifest.model_dump() | {"manifest_hash": HASH_A}))
