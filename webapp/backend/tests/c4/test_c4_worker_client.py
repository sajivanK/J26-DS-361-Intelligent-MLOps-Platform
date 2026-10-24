import importlib.util
import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

from fastapi import HTTPException

path = Path(__file__).resolve().parents[2] / "app/components/c4_worker_client.py"
spec = importlib.util.spec_from_file_location("worker_client_under_test", path)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class WorkerClientTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"C4_WORKER_URL": "http://127.0.0.1:8004"})
        env.start()
        self.addCleanup(env.stop)

    def opener(self, body=b'{"ok": true}'):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = body
        return opener

    def test_evaluate_payload(self):
        opener = self.opener()
        with patch.object(client, "build_opener", return_value=opener):
            result = client.evaluate("real-outer-case", "linear", "global")
        self.assertEqual(result, {"ok": True})
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.full_url,
                         "http://127.0.0.1:8004/api/c4/release/evaluate")
        self.assertEqual(json.loads(request.data), {
            "case_id": "real-outer-case", "method": "linear", "family": "global"
        })

    def test_catalog_get(self):
        opener = self.opener()
        with patch.object(client, "build_opener", return_value=opener):
            client.catalog()
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.full_url, "http://127.0.0.1:8004/api/c4/info")

    def test_connection_failure(self):
        opener = self.opener()
        opener.open.side_effect = URLError("connection refused")
        with patch.object(client, "build_opener", return_value=opener):
            with self.assertRaises(client.C4Unavailable):
                client.catalog()

    def test_http_errors(self):
        for status, expected, mapped in (
            (404, KeyError, None),
            (422, HTTPException, 422),
            (503, client.C4Unavailable, None),
            (500, HTTPException, 502),
        ):
            with self.subTest(status=status):
                opener = self.opener()
                opener.open.side_effect = HTTPError(
                    "http://127.0.0.1:8004", status, "error", {}, None
                )
                with patch.object(client, "build_opener", return_value=opener):
                    with self.assertRaises(expected) as caught:
                        client.catalog()
                if mapped is not None:
                    self.assertEqual(caught.exception.status_code, mapped)

    def test_invalid_json(self):
        with patch.object(client, "build_opener",
                          return_value=self.opener(b"invalid")):
            with self.assertRaises(client.C4Unavailable):
                client.catalog()

    def test_non_object_json(self):
        with patch.object(client, "build_opener",
                          return_value=self.opener(b"[]")):
            with self.assertRaises(client.C4Unavailable):
                client.catalog()

    def test_remote_address_rejected(self):
        with patch.dict(os.environ, {"C4_WORKER_URL": "http://example.com:8004"}):
            with patch.object(client, "build_opener") as opener:
                with self.assertRaises(client.C4Unavailable):
                    client.catalog()
                opener.assert_not_called()


if __name__ == "__main__":
    unittest.main()
