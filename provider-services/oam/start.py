#!/usr/bin/env python3
"""Start one fresh isolated OAM deployment without modifying existing providers."""

import argparse
import json
import os
import secrets
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

from oamb.runtime.provider_env import load_t10_provider_environment


def container_url(value):
    parts = urlsplit(value)
    if parts.hostname in {"127.0.0.1", "localhost", "::1"}:
        host = "host.docker.internal" + (f":{parts.port}" if parts.port else "")
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    return value


def service_environment(env, database, api, queue, content, root):
    return {
        "OAM_MIGRATION_DSN": f"postgresql://postgres:{database}@postgres:5432/oam",
        "OAM_API_DSN": f"postgresql://oamb_api:{api}@postgres:5432/oam",
        "OAM_WORKER_QUEUE_DSN": f"postgresql://oamb_queue:{queue}@postgres:5432/oam",
        "OAM_WORKER_CONTENT_DSN": f"postgresql://oamb_content:{content}@postgres:5432/oam",
        "OAM_ROOT_API_KEY": root,
        "VLM_BASE_URL": container_url(env["LLM_BASE_URL"]),
        "VLM_API_KEY": env["LLM_API_KEY"], "VLM_MODEL": env["LLM_LIGHT_MODEL"],
        "OAM_EMBEDDING_URL": container_url(env["OAMB_EMBEDDING_BASE_URL"]),
        "OAM_EMBEDDING_MODEL": env["OAMB_EMBEDDING_MODEL"],
        "OAM_EMBEDDING_KEY": env.get("OAMB_EMBEDDING_API_KEY") or "local-no-key",
    }


def write_env(path, values):
    if any("\n" in value or "\r" in value for value in values.values()):
        raise ValueError("environment values must be single-line")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write("".join(f"{key}={value}\n" for key, value in values.items()))


def ensure_fresh_project(command, project):
    existing = subprocess.run([*command, "ps", "-a", "-q"], check=True,
                              capture_output=True, text=True, timeout=30)
    volumes = subprocess.run(["docker", "volume", "ls", "--filter",
                              f"name=^{project}_postgres-data$", "-q"],
                             check=True, capture_output=True, text=True, timeout=30)
    if existing.stdout.strip() or volumes.stdout.strip():
        raise ValueError("compose project containers or data volume already exist; choose a fresh scope")


def wait_ready(base_url, *, timeout=60):
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=5, trust_env=False) as http:
        while True:
            try:
                response = http.get(base_url + "/health/ready")
                if response.status_code == 200 and response.json().get("ready") is True:
                    return
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise ValueError("OAM API readiness timed out; inspect the retained Compose logs")
            time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("lme500", "beam10m"), required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--project", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error("output already exists; choose a fresh scope")
    keys = frozenset({"LLM_BASE_URL", "LLM_API_KEY", "LLM_LIGHT_MODEL", "LLM_DEEP_MODEL",
                      "OAMB_EMBEDDING_BASE_URL", "OAMB_EMBEDDING_MODEL", "OAMB_EMBEDDING_API_KEY"})
    env = {**load_t10_provider_environment(args.env_file, expected_keys=keys), **os.environ}
    if any(not env.get(key) or env[key].startswith("change-me") for key in keys - {"OAMB_EMBEDDING_API_KEY"}):
        parser.error("configure generic LLM and embedding variables, including OAMB_EMBEDDING_MODEL")
    output.mkdir(parents=True, mode=0o700)
    passwords = [secrets.token_hex(24) for _ in range(5)]
    service = service_environment(env, *passwords)
    write_env(output / "service.env", service)
    directory = Path(__file__).resolve().parent
    compose_env = {"OAMB_OAM_IMAGE": args.image, "OAMB_OAM_DB_PASSWORD": passwords[0],
                   "OAMB_OAM_SERVICE_ENV": str(output / "service.env"),
                   "OAMB_OAM_CONFIG": str(directory / f"{args.dataset}.yaml")}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        compose_env["OAMB_OAM_PORT"] = str(listener.getsockname()[1])
    write_env(output / "compose.env", compose_env)
    command = ["docker", "compose", "--project-name", args.project, "--env-file", str(output / "compose.env"),
               "-f", str(directory / "compose.yaml")]
    def invoke(*arguments, **kwargs):
        return subprocess.run([*command, *arguments], check=True, timeout=300, **kwargs)
    # Check project freshness before any Docker mutation. Existing data is never removed.
    ensure_fresh_project(command, args.project)
    invoke("up", "-d", "--wait", "postgres")
    invoke("run", "--no-deps", "api", "migrate", "--config", "/opt/oam-eval/config.yaml")
    tenant = invoke("exec", "-T", "postgres", "psql", "-U", "postgres", "-d", "oam", "-tAc",
                    "SELECT tenant_id FROM bootstrap_state WHERE singleton=true", capture_output=True, text=True).stdout.strip()
    if not tenant:
        raise ValueError("bootstrap tenant is unavailable")
    sql = (directory / "initialize.sql").read_text()
    for key, value in zip(("api", "queue", "content"), passwords[1:4], strict=True):
        sql = sql.replace(f":'{key}_password'", f"'{value}'")
    invoke("exec", "-T", "postgres", "psql", "-U", "postgres", "-d", "oam", "-v", "ON_ERROR_STOP=1",
           input=sql, text=True, stdout=subprocess.DEVNULL)
    invoke("up", "-d", "api", "worker")
    port = invoke("port", "api", "8080", capture_output=True, text=True).stdout.strip()
    container = invoke("ps", "-q", "api", capture_output=True, text=True).stdout.strip()
    client = {key: env.get(key, "") for key in keys}
    client.update(OAMB_OAM_BASE_URL=f"http://{port}", OAMB_OAM_ROOT_KEY=passwords[4], OAMB_OAM_CONTAINER=container)
    client["OAMB_OAM_TENANT_ID"] = tenant
    wait_ready(client["OAMB_OAM_BASE_URL"])
    write_env(output / "client.env", client)
    (output / "service.json").write_text(json.dumps({"project": args.project, "container": container,
        "url": client["OAMB_OAM_BASE_URL"], "image": args.image, "dataset": args.dataset}, indent=2))
    print(f"OAM isolated service started; client environment: {output / 'client.env'}")


if __name__ == "__main__":
    main()
