from __future__ import annotations

import importlib.util
import json
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

INSPECTOR_PATH = Path(__file__).resolve().parents[1] / "mem0" / "inspector.py"


def load_inspector():
    spec = importlib.util.spec_from_file_location("oamb_mem0_inspector", INSPECTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load inspector module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeDatabase:
    def __init__(self) -> None:
        self.table_exists = True
        self.count = 0
        self.rows: list[tuple[str, object]] = []
        self.calls: list[tuple[str, tuple[object, ...] | None]] = []
        self.failure: Exception | None = None

    def connect(self, _config):
        if self.failure is not None:
            raise self.failure
        return FakeConnection(self)


class FakeConnection:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def cursor(self):
        return FakeCursor(self.database)


class FakeCursor:
    def __init__(self, database: FakeDatabase) -> None:
        self.database = database
        self.one: tuple[object, ...] | None = None
        self.all: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: tuple[object, ...] | None = None) -> None:
        statement = " ".join(query.split())
        self.database.calls.append((statement, params))
        self.one = None
        self.all = []
        if statement.startswith("SELECT to_regclass"):
            self.one = ("oamb_memories" if self.database.table_exists else None,)
        elif statement.startswith("SELECT count(*)"):
            self.one = (self.database.count,)
        elif statement.startswith("SELECT id::text, payload"):
            cursor = params[1] if params is not None and len(params) == 3 else None
            limit = params[-1] if params is not None else 0
            rows = self.database.rows
            if cursor is not None:
                rows = [row for row in rows if row[0] > cursor]
            self.all = rows[: int(limit)]

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.all


class InspectorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_inspector()

    def config(self):
        return self.module.InspectorConfig(
            listen_host="127.0.0.1",
            listen_port=0,
            api_key="inspector-secret",
            postgres_host="mem0-postgres",
            postgres_port=5432,
            postgres_database="postgres",
            postgres_user="oamb_mem0",
            postgres_password="database-secret",
        )

    @contextmanager
    def running_inspector(self, database: FakeDatabase):
        server = self.module.create_server(self.config(), connection_factory=database.connect)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}"
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def request(
        self,
        base_url: str,
        path: str,
        *,
        key: str = "inspector-secret",
        method: str = "GET",
    ):
        request = Request(
            base_url + path,
            method=method,
            headers={"Authorization": f"Bearer {key}"},
        )
        with urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read())

    def test_health_does_not_contact_postgres(self) -> None:
        database = FakeDatabase()
        with self.running_inspector(database) as base_url:
            status, payload = self.request(base_url, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "mode": "read_only_projection"})
        self.assertEqual(database.calls, [])

    def test_missing_table_is_exact_empty_projection(self) -> None:
        database = FakeDatabase()
        database.table_exists = False
        run_id = "a" * 64
        with self.running_inspector(database) as base_url:
            status, payload = self.request(base_url, f"/v1/projection?run_id={run_id}")
        self.assertEqual(status, 200)
        self.assertEqual(
            payload,
            {
                "collection": "oamb_memories",
                "run_id": run_id,
                "count": 0,
                "points": [],
                "next_cursor": None,
            },
        )

    def test_projection_uses_fixed_table_parameterized_filter_and_no_vector(self) -> None:
        database = FakeDatabase()
        database.count = 2
        run_id = "b" * 64
        database.rows = [
            (
                "00000000-0000-0000-0000-000000000001",
                {"run_id": run_id, "memory": "one"},
            ),
            (
                "00000000-0000-0000-0000-000000000002",
                {"run_id": run_id, "memory": "two"},
            ),
        ]
        with self.running_inspector(database) as base_url:
            status, payload = self.request(base_url, f"/v1/projection?run_id={run_id}")
        self.assertEqual(status, 200)
        self.assertEqual(payload["count"], 2)
        self.assertEqual(
            [point["id"] for point in payload["points"]],
            [row[0] for row in database.rows],
        )
        statements = [statement for statement, _params in database.calls]
        self.assertIn("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY", statements)
        self.assertTrue(
            any("SELECT set_config('statement_timeout'" in statement for statement in statements)
        )
        count_call = next(call for call in database.calls if call[0].startswith("SELECT count(*)"))
        page_call = next(
            call for call in database.calls if call[0].startswith("SELECT id::text, payload")
        )
        self.assertIn("FROM public.oamb_memories", count_call[0])
        self.assertIn("FROM public.oamb_memories", page_call[0])
        self.assertNotIn("vector", page_call[0].lower())
        self.assertEqual(count_call[1], (run_id,))
        self.assertEqual(page_call[1], (run_id, 257))

    def test_uuid_keyset_cursor_is_opaque_and_bounded(self) -> None:
        database = FakeDatabase()
        run_id = "c" * 64
        database.rows = [
            (f"00000000-0000-0000-0000-{index:012x}", {"run_id": run_id}) for index in range(1, 258)
        ]
        database.count = len(database.rows)
        with self.running_inspector(database) as base_url:
            _, first = self.request(base_url, f"/v1/projection?run_id={run_id}")
            self.assertEqual(len(first["points"]), 256)
            cursor = first["next_cursor"]
            self.assertIsInstance(cursor, str)
            self.assertNotIn(database.rows[255][0], cursor)
            _, second = self.request(base_url, f"/v1/projection?run_id={run_id}&cursor={cursor}")
        self.assertEqual(
            [point["id"] for point in second["points"]],
            [database.rows[256][0]],
        )
        page_calls = [
            call for call in database.calls if call[0].startswith("SELECT id::text, payload")
        ]
        self.assertEqual(page_calls[-1][1], (run_id, database.rows[255][0], 257))

    def test_client_errors_keep_bounded_4xx_contract(self) -> None:
        database = FakeDatabase()
        with self.running_inspector(database) as base_url:
            cases = (
                ("/v1/projection", "inspector-secret", "GET", 400),
                ("/v1/projection?run_id=short", "inspector-secret", "GET", 400),
                (
                    f"/v1/projection?run_id={'d' * 64}&cursor=bad!",
                    "inspector-secret",
                    "GET",
                    400,
                ),
                (f"/v1/projection?run_id={'e' * 64}", "wrong", "GET", 401),
                (
                    f"/v1/projection?run_id={'f' * 64}",
                    "inspector-secret",
                    "POST",
                    405,
                ),
            )
            for path, key, method, expected in cases:
                with self.subTest(path=path, expected=expected):
                    with self.assertRaises(HTTPError) as error:
                        self.request(base_url, path, key=key, method=method)
                    self.assertEqual(error.exception.code, expected)
                    error.exception.close()

    def test_invalid_database_results_fail_closed_as_generic_502(self) -> None:
        run_id = "1" * 64
        invalid_databases = []
        negative_count = FakeDatabase()
        negative_count.count = -1
        invalid_databases.append(negative_count)
        count_mismatch = FakeDatabase()
        count_mismatch.count = 1
        invalid_databases.append(count_mismatch)
        invalid_payload = FakeDatabase()
        invalid_payload.count = 1
        invalid_payload.rows = [("00000000-0000-0000-0000-000000000001", [])]
        invalid_databases.append(invalid_payload)
        duplicate = FakeDatabase()
        duplicate.count = 2
        duplicate.rows = [
            ("00000000-0000-0000-0000-000000000001", {"run_id": run_id}),
            ("00000000-0000-0000-0000-000000000001", {"run_id": run_id}),
        ]
        invalid_databases.append(duplicate)
        timed_out = FakeDatabase()
        timed_out.failure = TimeoutError("database timeout")
        invalid_databases.append(timed_out)

        for database in invalid_databases:
            with self.subTest(database=database):
                with self.running_inspector(database) as base_url:
                    with self.assertRaises(HTTPError) as error:
                        self.request(base_url, f"/v1/projection?run_id={run_id}")
                    self.assertEqual(error.exception.code, 502)
                    self.assertEqual(
                        json.loads(error.exception.read()),
                        {"error": "PostgreSQL projection unavailable"},
                    )
                    error.exception.close()

    def test_oversize_response_fails_closed(self) -> None:
        database = FakeDatabase()
        run_id = "2" * 64
        database.count = 1
        database.rows = [
            (
                "00000000-0000-0000-0000-000000000001",
                {"run_id": run_id, "memory": "x" * 256},
            )
        ]
        original_limit = self.module.MAX_INSPECTOR_RESPONSE_BYTES
        self.module.MAX_INSPECTOR_RESPONSE_BYTES = 128
        try:
            with self.running_inspector(database) as base_url:
                with self.assertRaises(HTTPError) as error:
                    self.request(base_url, f"/v1/projection?run_id={run_id}")
                self.assertEqual(error.exception.code, 502)
                error.exception.close()
        finally:
            self.module.MAX_INSPECTOR_RESPONSE_BYTES = original_limit


if __name__ == "__main__":
    unittest.main()
