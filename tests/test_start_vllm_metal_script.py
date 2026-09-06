from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from tests.benchmark_configuration import load_canonical_configuration

SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "start_local_embedding" / "start_vllm_metal.sh"
)
EMBEDDING_MODEL = load_canonical_configuration().models.embedding.model


def test_start_vllm_metal_is_executable_from_readme_command() -> None:
    assert os.access(SCRIPT_PATH, os.X_OK)


def _write_vllm(executable: Path, body: str) -> None:
    executable.parent.mkdir(parents=True)
    executable.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    executable.chmod(0o755)


def _copy_script(tmp_path: Path) -> Path:
    script = (
        tmp_path
        / "open-agent-memory-benchmark"
        / "scripts"
        / "start_local_embedding"
        / SCRIPT_PATH.name
    )
    script.parent.mkdir(parents=True)
    shutil.copy2(SCRIPT_PATH, script)
    return script


def _write_cached_embedding_model(home: Path) -> tuple[Path, Path]:
    model_path = home / ".cache" / "qmd" / "models" / "Qwen3-Embedding-0.6B-Q8_0.gguf"
    model_path.parent.mkdir(parents=True)
    model_path.write_text("cached gguf", encoding="utf-8")
    hf_dir = home / ".cache" / "oamb-vllm-metal" / "Qwen3-Embedding-0.6B-hf"
    hf_dir.mkdir(parents=True)
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return model_path, hf_dir


def test_start_vllm_metal_uses_sibling_environment_not_shell_path(tmp_path: Path) -> None:
    script = _copy_script(tmp_path)
    capture = tmp_path / "invocation.txt"
    home = tmp_path / "home"
    model_path, hf_dir = _write_cached_embedding_model(home)
    metal_vllm = tmp_path / "vllm-metal" / ".venv-vllm-metal" / "bin" / "vllm"
    fallback_vllm = tmp_path / "fallback" / "vllm"
    _write_vllm(
        metal_vllm,
        'printf "%s\\n%s\\n%s\\n" "$VLLM_METAL_BUILD_FROM_SOURCE" '
        '"$VLLM_METAL_MEMORY_FRACTION" "$*" > "$CAPTURE"',
    )
    _write_vllm(fallback_vllm, 'printf "fallback\\n" > "$CAPTURE"')

    result = subprocess.run(
        ["/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CAPTURE": str(capture),
            "HOME": str(home),
            "OAMB_EMBEDDING_MODEL": EMBEDDING_MODEL,
            "PATH": f"{fallback_vllm.parent}:/usr/bin:/bin",
        },
    )

    assert result.returncode == 0, result.stderr
    lines = capture.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == ["1", "0.06"]
    assert lines[2] == (
        f"serve {model_path} --tokenizer {hf_dir} --hf-config-path {hf_dir} "
        '--hf-overrides {"matryoshka_dimensions":[1024]} --runner pooling '
        "--max-model-len 8192 --max-num-batched-tokens 8192 --additional-config "
        '{"turboquant":true,"k_quant":"q8_0","v_quant":"q8_0"} '
        f"--host 127.0.0.1 --port 18000 --served-model-name {EMBEDDING_MODEL}"
    )


def test_start_vllm_metal_rejects_missing_environment_without_path_fallback(
    tmp_path: Path,
) -> None:
    script = _copy_script(tmp_path)
    capture = tmp_path / "fallback.txt"
    fallback_vllm = tmp_path / "fallback" / "vllm"
    _write_vllm(fallback_vllm, 'printf "called\\n" > "$CAPTURE"')

    result = subprocess.run(
        ["/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CAPTURE": str(capture),
            "HOME": str(tmp_path / "home"),
            "OAMB_EMBEDDING_MODEL": EMBEDDING_MODEL,
            "PATH": f"{fallback_vllm.parent}:/usr/bin:/bin",
        },
    )

    assert result.returncode != 0
    assert "vLLM-Metal executable not found" in result.stderr
    assert not capture.exists()


def test_start_vllm_metal_rejects_missing_cached_model(tmp_path: Path) -> None:
    script = _copy_script(tmp_path)
    capture = tmp_path / "invocation.txt"
    metal_vllm = tmp_path / "vllm-metal" / ".venv-vllm-metal" / "bin" / "vllm"
    _write_vllm(metal_vllm, 'printf "called\\n" > "$CAPTURE"')

    result = subprocess.run(
        ["/bin/bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CAPTURE": str(capture),
            "HOME": str(tmp_path / "home"),
            "OAMB_EMBEDDING_MODEL": EMBEDDING_MODEL,
            "PATH": "/usr/bin:/bin",
        },
    )

    assert result.returncode != 0
    assert "cached Qwen3 embedding GGUF not found" in result.stderr
    assert not capture.exists()
