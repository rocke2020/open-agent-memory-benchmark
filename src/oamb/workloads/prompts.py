"""Strict byte-preserving PromptPack loading, rendering, and export evidence."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from oamb.contracts.ports import RenderedPrompt
from oamb.contracts.specifications import PromptPackManifest

_PLACEHOLDER = re.compile(rb"\{\{([a-z][a-z0-9_]*)\}\}")

ALL_PROMPT_PACK_IDS = (
    "oamb-lme-answer-v1",
    "oamb-lme-judge-v1",
    "oamb-mab-eventqa-rag-v1",
    "oamb-mab-icl-rag-v1",
    "oamb-mab-redial-rag-v1",
    "oamb-mab-detectiveqa-rag-v1",
    "oamb-mab-factconsolidation-rag-v1",
)


@dataclass(frozen=True, slots=True)
class PromptPack:
    manifest: PromptPackManifest
    templates: Mapping[str, bytes]

    def __post_init__(self) -> None:
        templates = dict(self.templates)
        expected_names = tuple(item.template_name for item in self.manifest.templates)
        if set(templates) != set(expected_names):
            raise ValueError("PromptPack templates must exactly match the manifest")
        for template in self.manifest.templates:
            content = templates[template.template_name]
            if len(content) != template.byte_count:
                raise ValueError(
                    f"PromptPack template byte count differs: {template.template_name}"
                )
            if hashlib.sha256(content).hexdigest() != template.content_sha256:
                raise ValueError(f"PromptPack template hash differs: {template.template_name}")
        object.__setattr__(self, "templates", MappingProxyType(templates))


def render_prompt(
    pack: PromptPack,
    template_name: str,
    variables: Mapping[str, str],
) -> RenderedPrompt:
    """Render exact UTF-8 values without newline or Unicode normalization."""

    expected = pack.manifest.variables
    if set(variables) != set(expected) or len(variables) != len(expected):
        raise ValueError("PromptPack variables must match the declared set exactly")
    if not all(isinstance(value, str) for value in variables.values()):
        raise TypeError("PromptPack variables must be strings")
    try:
        template = pack.templates[template_name]
    except KeyError as exc:
        raise ValueError(f"unknown PromptPack template: {template_name}") from exc
    used = tuple(match.group(1).decode("ascii") for match in _PLACEHOLDER.finditer(template))
    if set(used) != set(expected):
        raise ValueError("template placeholders must exactly cover declared variables")

    rendered = _PLACEHOLDER.sub(
        lambda match: variables[match.group(1).decode("ascii")].encode("utf-8"),
        template,
    )
    return RenderedPrompt(
        canonical_bytes=rendered,
        sha256=hashlib.sha256(rendered).hexdigest(),
        prompt_pack_id=pack.manifest.prompt_pack_id,
        template_name=template_name,
        variable_names=expected,
        variable_values=tuple(variables[name] for name in expected),
        variable_value_sha256=tuple(
            hashlib.sha256(variables[name].encode()).hexdigest() for name in expected
        ),
    )


def export_prompt_evidence(
    pack: PromptPack,
    rendered: RenderedPrompt,
    *,
    public: bool,
) -> dict[str, object]:
    if (
        rendered.prompt_pack_id != pack.manifest.prompt_pack_id
        or rendered.template_name is None
        or rendered.template_name not in pack.templates
    ):
        raise ValueError("rendered prompt does not belong to the PromptPack")
    if hashlib.sha256(rendered.canonical_bytes).hexdigest() != rendered.sha256:
        raise ValueError("rendered prompt hash does not match its bytes")
    if (
        rendered.variable_names != pack.manifest.variables
        or len(rendered.variable_values) != len(pack.manifest.variables)
        or len(rendered.variable_value_sha256) != len(pack.manifest.variables)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", value_hash) is None
            for value_hash in rendered.variable_value_sha256
        )
    ):
        raise ValueError("rendered prompt variables do not match the PromptPack")
    derived = render_prompt(
        pack,
        rendered.template_name,
        dict(zip(rendered.variable_names, rendered.variable_values, strict=True)),
    )
    if derived != rendered:
        raise ValueError("rendered prompt was not derived from the exact PromptPack variables")
    template = pack.templates[rendered.template_name]
    evidence: dict[str, object] = {
        "prompt_pack_id": pack.manifest.prompt_pack_id,
        "manifest_sha256": pack.manifest.manifest_sha256,
        "template_name": rendered.template_name,
        "template_sha256": hashlib.sha256(template).hexdigest(),
        "template_byte_count": len(template),
        "rendered_sha256": rendered.sha256,
        "rendered_byte_count": len(rendered.canonical_bytes),
        "variable_names": rendered.variable_names,
        "variable_value_sha256": rendered.variable_value_sha256,
        "redistribution_allowed": pack.manifest.redistribution_allowed,
    }
    if not public:
        evidence["template_text"] = template.decode("utf-8")
        evidence["rendered_text"] = rendered.canonical_bytes.decode("utf-8")
    elif pack.manifest.redistribution_allowed:
        evidence["template_text"] = template.decode("utf-8")
    return evidence
