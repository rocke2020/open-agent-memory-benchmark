from __future__ import annotations

from pathlib import Path

import pytest


def test_loads_only_expected_provider_values_as_inert_text(tmp_path: Path) -> None:
    from oamb.runtime.provider_env import load_t10_provider_environment

    marker = tmp_path / "must-not-exist"
    env_file = tmp_path / ".env"
    hostile = f"literal;touch {marker}"
    env_file.write_text(
        "\n".join(
            (
                "# provider configuration remains inert data",
                "OAMB_MEM0_PORT=18889",
                f"OAMB_MEM0_ADMIN_API_KEY={hostile}",
                "OAMB_MEM0_INSPECTOR_API_KEY=inspector-key",
                "OAMB_MEM0_JWT_SECRET=ignored-secret",
                "",
            )
        ),
        encoding="utf-8",
    )

    loaded = load_t10_provider_environment(
        env_file,
        expected_keys=frozenset(
            {
                "OAMB_MEM0_PORT",
                "OAMB_MEM0_ADMIN_API_KEY",
                "OAMB_MEM0_INSPECTOR_API_KEY",
            }
        ),
    )

    assert loaded == {
        "OAMB_MEM0_PORT": "18889",
        "OAMB_MEM0_ADMIN_API_KEY": hostile,
        "OAMB_MEM0_INSPECTOR_API_KEY": "inspector-key",
    }
    assert not marker.exists()


def test_default_profile_loads_selected_answer_and_judge_gateway_references(
    tmp_path: Path,
) -> None:
    from oamb.runtime.provider_env import load_t10_provider_environment

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OAMB_HINDSIGHT_LLM_BASE_URL=https://gateway.example.invalid/v1\n"
        "OAMB_MEM0_LLM_API_KEY=mem0-secret\n"
        "OAMB_MEM0_LLM_BASE_URL=https://gateway.example.invalid/v1\n"
        "OAMB_HINDSIGHT_LLM_API_KEY=benchmark-evaluation-key\n",
        encoding="utf-8",
    )

    assert load_t10_provider_environment(env_file) == {
        "OAMB_HINDSIGHT_LLM_BASE_URL": "https://gateway.example.invalid/v1",
        "OAMB_MEM0_LLM_API_KEY": "mem0-secret",
        "OAMB_MEM0_LLM_BASE_URL": "https://gateway.example.invalid/v1",
        "OAMB_HINDSIGHT_LLM_API_KEY": "benchmark-evaluation-key",
    }


def test_default_profile_selects_vllm_metal_endpoint_and_ignores_legacy_name(
    tmp_path: Path,
) -> None:
    from oamb.runtime.provider_env import load_t10_provider_environment

    env_file = tmp_path / ".env"
    env_file.write_text(
        "OAMB_EMBEDDING_BASE_URL=http://127.0.0.1:18000/v1\n"
        "OAMB_OLLAMA_BASE_URL=http://127.0.0.1:11434/v1\n",
        encoding="utf-8",
    )

    assert load_t10_provider_environment(env_file) == {
        "OAMB_EMBEDDING_BASE_URL": "http://127.0.0.1:18000/v1",
    }


def test_unselected_values_do_not_block_loading_selected_inert_data(tmp_path: Path) -> None:
    from oamb.runtime.provider_env import load_t10_provider_environment

    env_file = tmp_path / ".env"
    env_file.write_text(
        'OPENAI_BASE_URL="https://ignored.example/v1"\n'
        'OPENAI_API_KEY="ignored-value"\n'
        "DEEPSEEK_BASE_URL=https://selected.example/v1\n"
        "DEEPSEEK_API_KEY=selected-value\n",
        encoding="utf-8",
    )

    assert load_t10_provider_environment(
        env_file,
        expected_keys=frozenset({"DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"}),
    ) == {
        "DEEPSEEK_BASE_URL": "https://selected.example/v1",
        "DEEPSEEK_API_KEY": "selected-value",
    }


@pytest.mark.parametrize(
    ("content", "message"),
    (
        ("OAMB_MEM0_PORT=1\nOAMB_MEM0_PORT=2\n", "duplicate"),
        ("export OAMB_MEM0_PORT=18889\n", "key"),
        ("OAMB_MEM0_PORT =18889\n", "key"),
        ("OAMB_MEM0_PORT= 18889\n", "value syntax"),
        ('OAMB_MEM0_PORT="18889"\n', "value syntax"),
        ("OAMB_MEM0_PORT=$(touch /tmp/no)\n", "value syntax"),
        ("OAMB_MEM0_PORT=18889 # comment\n", "value syntax"),
    ),
)
def test_rejects_duplicate_or_unsupported_provider_dotenv_syntax(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    from oamb.runtime.provider_env import (
        ProviderEnvironmentFileError,
        load_t10_provider_environment,
    )

    env_file = tmp_path / ".env"
    env_file.write_text(content, encoding="utf-8")

    with pytest.raises(ProviderEnvironmentFileError, match=message):
        load_t10_provider_environment(
            env_file,
            expected_keys=frozenset({"OAMB_MEM0_PORT"}),
        )


def test_provider_dotenv_rejects_symlink_and_non_utf8(tmp_path: Path) -> None:
    from oamb.runtime.provider_env import (
        ProviderEnvironmentFileError,
        load_t10_provider_environment,
    )

    target = tmp_path / "target.env"
    target.write_text("OAMB_MEM0_PORT=18889\n", encoding="utf-8")
    link = tmp_path / "link.env"
    link.symlink_to(target)

    with pytest.raises(ProviderEnvironmentFileError, match="regular file"):
        load_t10_provider_environment(
            link,
            expected_keys=frozenset({"OAMB_MEM0_PORT"}),
        )

    invalid = tmp_path / "invalid.env"
    invalid.write_bytes(b"OAMB_MEM0_PORT=\xff\n")
    with pytest.raises(ProviderEnvironmentFileError, match="UTF-8"):
        load_t10_provider_environment(
            invalid,
            expected_keys=frozenset({"OAMB_MEM0_PORT"}),
        )
