#!/usr/bin/env python3
"""Provision one BEAM owner index in an already running isolated OAM container."""

import json
import subprocess
import sys

CONFIG = "/opt/oam-eval/config.yaml"
CHECK = (
    "from oam.config import load_config; "
    f"c=load_config({CONFIG!r}).config; "
    "assert c.database.vector_index_mode == 'large', 'BEAM requires large index mode'"
)


def main() -> int:
    if len(sys.argv) != 4:
        print("Usage: provision-beam-index.py CONTAINER TENANT_ID USER_ID", file=sys.stderr)
        return 2
    container, tenant, user = sys.argv[1:]
    try:
        subprocess.run(["docker", "exec", container, "python", "-c", CHECK], check=True)
        result = subprocess.run(
            [
                "docker",
                "exec",
                container,
                "oam",
                "vector-indexes",
                "--config",
                CONFIG,
                "--once",
                "--tenant-id",
                tenant,
                "--user-id",
                user,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(result.stdout)
        owners = report["owners"]
        if (
            report["busy"]
            or len(owners) != 1
            or owners[0].get("tenant_id") != tenant
            or owners[0].get("user_id") != user
            or owners[0].get("state") != "ready"
        ):
            raise ValueError(
                "owner index is not ready; ensure indexing produced embedded search units"
            )
        print(json.dumps(report))
        return 0
    except (subprocess.CalledProcessError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"BEAM index provisioning failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
