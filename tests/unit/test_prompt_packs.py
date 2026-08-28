from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, cast

import pytest

from oamb.contracts.ports import RenderedPrompt
from oamb.contracts.specifications import (
    PromptPackManifest,
    PromptTemplateManifest,
    prompt_pack_manifest_hash,
)
from oamb.workloads.prompts import PromptPack, export_prompt_evidence, render_prompt


def _pack(*, redistribution_allowed: bool) -> PromptPack:
    template = b"Question: {{question}}\nMemory: {{retrieved_context}}"
    values: Any = dict(
        prompt_pack_id="fixture-answer-v1",
        prompt_pack_version="1",
        workload_id="fixture-workload-v1",
        origin="user_supplied" if not redistribution_allowed else "oamb_authored",
        source_repository=None,
        source_revision=None,
        source_file_sha256=None,
        license_expression="NOASSERTION" if not redistribution_allowed else "Apache-2.0",
        rights_attestation=(
            "user_declared_authorized_use"
            if not redistribution_allowed
            else "project_licensed_source"
        ),
        redistribution_allowed=redistribution_allowed,
        templates=(
            PromptTemplateManifest(
                template_name="answer_user",
                relative_path="answer-user.txt",
                content_sha256=hashlib.sha256(template).hexdigest(),
                byte_count=len(template),
            ),
        ),
        variables=("question", "retrieved_context"),
        output_contract_id="lme-answer-text-v1",
    )
    manifest = PromptPackManifest(
        manifest_sha256=prompt_pack_manifest_hash(values),
        **values,
    )
    return PromptPack(manifest=manifest, templates={"answer_user": template})


def test_render_prompt_requires_the_exact_declared_variable_set_and_bytes() -> None:
    pack = _pack(redistribution_allowed=True)

    rendered = render_prompt(
        pack,
        "answer_user",
        {"question": "哪一个？", "retrieved_context": "alpha\r\nbeta"},
    )

    assert rendered.canonical_bytes == "Question: 哪一个？\nMemory: alpha\r\nbeta".encode()
    assert rendered.sha256 == hashlib.sha256(rendered.canonical_bytes).hexdigest()
    assert rendered.variable_names == ("question", "retrieved_context")
    with pytest.raises(ValueError, match="exactly"):
        render_prompt(pack, "answer_user", {"question": "missing context"})
    with pytest.raises(ValueError, match="exactly"):
        render_prompt(
            pack,
            "answer_user",
            {"question": "q", "retrieved_context": "c", "gold_answer": "secret"},
        )


def test_render_prompt_substitutes_only_template_placeholders_once() -> None:
    pack = _pack(redistribution_allowed=True)

    rendered = render_prompt(
        pack,
        "answer_user",
        {
            "question": "What does {{retrieved_context}} mean?",
            "retrieved_context": "literal {{question}} evidence",
        },
    )

    assert rendered.canonical_bytes == (
        b"Question: What does {{retrieved_context}} mean?\nMemory: literal {{question}} evidence"
    )


def test_prompt_pack_copies_and_freezes_validated_template_bytes() -> None:
    pack = _pack(redistribution_allowed=True)
    templates = cast(dict[str, bytes], pack.templates)

    with pytest.raises(TypeError):
        templates["answer_user"] = b"changed after manifest validation"

    rendered = render_prompt(
        pack,
        "answer_user",
        {"question": "question", "retrieved_context": "context"},
    )
    assert rendered.canonical_bytes == b"Question: question\nMemory: context"


def test_private_prompt_public_evidence_retains_hashes_but_never_text() -> None:
    pack = _pack(redistribution_allowed=False)
    rendered = render_prompt(
        pack,
        "answer_user",
        {"question": "private question", "retrieved_context": "private context"},
    )

    public = export_prompt_evidence(pack, rendered, public=True)
    local = export_prompt_evidence(pack, rendered, public=False)

    assert public["rendered_sha256"] == rendered.sha256
    assert public["rendered_byte_count"] == len(rendered.canonical_bytes)
    assert "rendered_text" not in public
    assert "template_text" not in public
    assert local["rendered_text"] == rendered.canonical_bytes.decode()
    assert local["template_text"] == pack.templates["answer_user"].decode()


def test_redistributable_template_never_publishes_private_rendered_variables() -> None:
    pack = _pack(redistribution_allowed=True)
    rendered = render_prompt(
        pack,
        "answer_user",
        {
            "question": "PRIVATE-DATASET-QUESTION",
            "retrieved_context": "PRIVATE-DATASET-CONTEXT",
        },
    )

    public = export_prompt_evidence(pack, rendered, public=True)
    local = export_prompt_evidence(pack, rendered, public=False)

    assert public["template_text"] == pack.templates["answer_user"].decode()
    assert "rendered_text" not in public
    assert "PRIVATE-DATASET" not in repr(public)
    local_rendered_text = local["rendered_text"]
    assert isinstance(local_rendered_text, str)
    assert "PRIVATE-DATASET-CONTEXT" in local_rendered_text


def test_prompt_export_rejects_tampered_or_foreign_rendered_evidence() -> None:
    pack = _pack(redistribution_allowed=True)
    rendered = render_prompt(
        pack,
        "answer_user",
        {"question": "question", "retrieved_context": "context"},
    )

    with pytest.raises(ValueError, match="hash"):
        export_prompt_evidence(
            pack,
            RenderedPrompt(
                canonical_bytes=rendered.canonical_bytes + b"!",
                sha256=rendered.sha256,
                prompt_pack_id=rendered.prompt_pack_id,
                template_name=rendered.template_name,
                variable_names=rendered.variable_names,
                variable_value_sha256=rendered.variable_value_sha256,
            ),
            public=True,
        )
    with pytest.raises(ValueError, match="PromptPack"):
        export_prompt_evidence(
            pack,
            RenderedPrompt(
                canonical_bytes=rendered.canonical_bytes,
                sha256=rendered.sha256,
                prompt_pack_id="different-pack",
                template_name=rendered.template_name,
                variable_names=rendered.variable_names,
                variable_value_sha256=rendered.variable_value_sha256,
            ),
            public=True,
        )


def test_prompt_export_rejects_self_hashed_bytes_not_derived_from_the_template() -> None:
    pack = _pack(redistribution_allowed=True)
    rendered = render_prompt(
        pack,
        "answer_user",
        {"question": "question", "retrieved_context": "context"},
    )
    forged_bytes = b"self-hashed bytes that the PromptPack cannot render"

    with pytest.raises(ValueError, match="derived"):
        export_prompt_evidence(
            pack,
            replace(
                rendered,
                canonical_bytes=forged_bytes,
                sha256=hashlib.sha256(forged_bytes).hexdigest(),
            ),
            public=True,
        )
