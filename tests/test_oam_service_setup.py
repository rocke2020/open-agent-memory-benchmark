import importlib.util
from pathlib import Path

import pytest


def test_service_setup_keeps_role_credentials_separate_and_rewrites_only_loopback():
    path = Path(__file__).parents[1] / "provider-services/oam/start.py"
    spec = importlib.util.spec_from_file_location("oam_start", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = {
        "LLM_BASE_URL": "https://model.example/v1",
        "LLM_API_KEY": "test-key",
        "LLM_LIGHT_MODEL": "model",
        "OAMB_EMBEDDING_BASE_URL": "http://127.0.0.1:8000/v1",
        "OAMB_EMBEDDING_MODEL": "embed",
        "OAMB_EMBEDDING_API_KEY": "embed-key",
    }
    result = module.service_environment(env, "db-secret", "api", "queue", "content", "root")
    assert result["OAM_EMBEDDING_URL"] == "http://host.docker.internal:8000/v1"
    assert result["VLM_BASE_URL"] == "https://model.example/v1"
    assert "oamb_api:api@postgres" in result["OAM_API_DSN"]
    assert "oamb_queue:queue@postgres" in result["OAM_WORKER_QUEUE_DSN"]
    assert "oamb_content:content@postgres" in result["OAM_WORKER_CONTENT_DSN"]


def test_setup_rejects_retained_volume_before_any_mutation(monkeypatch):
    import subprocess

    path = Path(__file__).parents[1] / "provider-services/oam/start.py"
    spec = importlib.util.spec_from_file_location("oam_start", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_run(args, **kwargs):
        assert "up" not in args and "run" not in args
        return subprocess.CompletedProcess(
            args, 0, stdout="project_postgres-data\n" if "volume" in args else ""
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    with pytest.raises(ValueError, match="volume"):
        module.ensure_fresh_project(["docker", "compose"], "project")


@pytest.mark.parametrize("becomes_ready", [True, False])
def test_setup_waits_for_api_readiness_or_fails_bounded(monkeypatch, becomes_ready):
    import httpx

    path = Path(__file__).parents[1] / "provider-services/oam/start.py"
    spec = importlib.util.spec_from_file_location("oam_start", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    attempts = []
    client = httpx.Client

    def handle(request):
        attempts.append(request.url.path)
        if len(attempts) == 1:
            raise httpx.ConnectError("still starting", request=request)
        return httpx.Response(200, json={"ready": becomes_ready})

    monkeypatch.setattr(
        module.httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handle))
    )
    ticks = iter([0, 0, 1, 2])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    if becomes_ready:
        module.wait_ready("http://oam", timeout=2)
    else:
        with pytest.raises(ValueError, match="readiness"):
            module.wait_ready("http://oam", timeout=2)
    assert len(attempts) >= 2
