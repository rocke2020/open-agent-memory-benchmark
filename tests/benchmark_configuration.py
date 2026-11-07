"""Test-only workload variants derived from the canonical benchmark config."""

from __future__ import annotations

import tempfile
from pathlib import Path

from oamb.config.benchmark import BenchmarkConfiguration, load_benchmark_configuration

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "benchmark.yml"
MODEL_ENVIRONMENT = {
    "LLM_LIGHT_MODEL": "deepseek-flash",
    "LLM_DEEP_MODEL": "deepseek-flash",
}


def load_canonical_configuration() -> BenchmarkConfiguration:
    return load_benchmark_configuration(
        BENCHMARK_CONFIG_PATH,
        model_environment=MODEL_ENVIRONMENT,
    )


def lme6_configuration_text() -> str:
    """Derive the small regression workload without copying model configuration."""

    content = BENCHMARK_CONFIG_PATH.read_text(encoding="utf-8")
    replacements = (
        (
            "# Frozen benchmark profile configuration for one complete v0.1 LME-60 evaluation.",
            "# Internal LME-6 regression fixture.",
        ),
        ("comparison: v0.1-lme60", "comparison: t10-lme6"),
        ("workload_id: lme60-balanced-v1", "workload_id: lme30-native-smoke-plus-v1"),
        ("selection: lme60", "selection: lme6"),
        (
            "case_manifest_hash: 20afb9ae0d18a5821871e76a1b90df573fd5f155511c264249fdb49154088962",
            "case_manifest_hash: 0462b0a8735597d206c2219f174c36910c60dc11e89b47cc56ca9326048229ab",
        ),
        ("cell_id: hindsight-lme60", "cell_id: hindsight-lme6"),
        ("cell_id: mem0-lme60", "cell_id: mem0-lme6"),
        ("cell_id: openviking-lme60", "cell_id: openviking-lme6"),
        (
            content[content.index("\ndecision:\n") :],
            "",
        ),
    )
    for original, replacement in replacements:
        if content.count(original) != 1:
            raise AssertionError(f"canonical benchmark fragment is not unique: {original}")
        content = content.replace(original, replacement, 1)
    return content


def write_lme6_configuration(path: Path) -> Path:
    path.write_text(lme6_configuration_text(), encoding="utf-8")
    return path


def load_lme6_configuration() -> BenchmarkConfiguration:
    with tempfile.TemporaryDirectory() as directory:
        path = write_lme6_configuration(Path(directory) / "benchmark-lme6.yml")
        return load_benchmark_configuration(path, model_environment=MODEL_ENVIRONMENT)
