from __future__ import annotations

import importlib.util
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


class FakeQdrantHandler(BaseHTTPRequestHandler):
    calls: list[tuple[str, str, dict[str, object]]] = []
    valid_response = True

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        self.calls.append(("GET", self.path, {}))
        payload = {"status": "ok", "result": {"points_count": 2, "vectors_count": 2}}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        self.calls.append(("POST", self.path, body))
        if self.path.endswith("/points/count"):
            result: dict[str, object] = {"count": 2}
        else:
            result = {"points": [], "next_page_offset": None}
        payload = {"status": "ok", "result": result} if self.valid_response else {"unexpected": True}
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())


class InspectorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_inspector()
        FakeQdrantHandler.calls = []
        cls.qdrant = ThreadingHTTPServer(("127.0.0.1", 0), FakeQdrantHandler)
        cls.qdrant_thread = threading.Thread(target=cls.qdrant.serve_forever, daemon=True)
        cls.qdrant_thread.start()
        config = cls.module.InspectorConfig(
            listen_host="127.0.0.1",
            listen_port=0,
            api_key="inspector-secret",
            qdrant_url=f"http://127.0.0.1:{cls.qdrant.server_port}",
            qdrant_api_key="backend-secret",
            collection="oamb_memories",
        )
        cls.inspector = cls.module.create_server(config)
        cls.inspector_thread = threading.Thread(
            target=cls.inspector.serve_forever, daemon=True
        )
        cls.inspector_thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.inspector.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.inspector.shutdown()
        cls.inspector.server_close()
        cls.qdrant.shutdown()
        cls.qdrant.server_close()

    def request(self, path: str, *, key: str = "inspector-secret", method: str = "GET"):
        request = Request(
            self.base_url + path,
            method=method,
            headers={"Authorization": f"Bearer {key}"},
        )
        with urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read())

    def test_health_does_not_contact_qdrant(self) -> None:
        before = len(FakeQdrantHandler.calls)
        status, payload = self.request("/health")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok", "mode": "read_only_projection"})
        self.assertEqual(len(FakeQdrantHandler.calls), before)

    def test_projection_builds_exact_run_id_filter(self) -> None:
        run_id = "a" * 64
        status, payload = self.request(f"/v1/projection?run_id={run_id}")
        self.assertEqual(status, 200)
        self.assertEqual(payload["run_id"], run_id)
        calls = FakeQdrantHandler.calls[-2:]
        self.assertEqual(calls[0][1], "/collections/oamb_memories/points/count")
        self.assertEqual(calls[1][1], "/collections/oamb_memories/points/scroll")
        expected_filter = {"must": [{"key": "run_id", "match": {"value": run_id}}]}
        self.assertEqual(calls[0][2]["filter"], expected_filter)
        self.assertEqual(calls[1][2]["filter"], expected_filter)
        self.assertFalse(calls[1][2]["with_vector"])

    def test_unscoped_malformed_and_unauthorized_requests_fail(self) -> None:
        for path, key, expected in (
            ("/v1/projection", "inspector-secret", 400),
            ("/v1/projection?run_id=short", "inspector-secret", 400),
            (f"/v1/projection?run_id={'b' * 64}", "wrong", 401),
            ("/v1/collections/other", "inspector-secret", 404),
        ):
            with self.subTest(path=path, expected=expected):
                with self.assertRaises(HTTPError) as error:
                    self.request(path, key=key)
                self.assertEqual(error.exception.code, expected)
                error.exception.close()

    def test_write_methods_are_not_representable(self) -> None:
        with self.assertRaises(HTTPError) as error:
            self.request(f"/v1/projection?run_id={'c' * 64}", method="POST")
        self.assertEqual(error.exception.code, 405)
        error.exception.close()

    def test_invalid_backend_response_is_502_not_client_400(self) -> None:
        FakeQdrantHandler.valid_response = False
        try:
            with self.assertRaises(HTTPError) as error:
                self.request(f"/v1/projection?run_id={'d' * 64}")
            self.assertEqual(error.exception.code, 502)
            error.exception.close()
        finally:
            FakeQdrantHandler.valid_response = True


if __name__ == "__main__":
    unittest.main()
