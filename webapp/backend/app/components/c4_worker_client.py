"""Forward C4 requests without importing C4 model libraries."""
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, ProxyHandler, build_opener

from fastapi import HTTPException


class C4Unavailable(RuntimeError):
    pass


def _request(path, payload=None):
    base = os.environ.get("C4_WORKER_URL", "http://127.0.0.1:8004").rstrip("/")
    parsed = urlsplit(base)
    if (parsed.scheme != "http"
            or parsed.hostname not in ("127.0.0.1", "localhost")
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment):
        raise C4Unavailable("C4_WORKER_URL must be a local HTTP worker address.")

    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    request = Request(
        base + path, data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    # Local communication should not pass through system HTTP proxies.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(request, timeout=300) as response:
            body = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            raise KeyError("Unknown recorded C4 case.") from exc
        if exc.code == 422:
            raise HTTPException(422, "C4 worker rejected the request.") from exc
        if exc.code == 503:
            raise C4Unavailable("C4 worker artifacts are unavailable.") from exc
        raise HTTPException(502, "C4 worker failed. Inspect its terminal.") from exc
    except (URLError, OSError) as exc:
        raise C4Unavailable("Cannot reach the C4 worker on port 8004. Keep its terminal running.") from exc

    try:
        result = json.loads(body)
    except (ValueError, UnicodeError) as exc:
        raise C4Unavailable("C4 worker returned an invalid response.") from exc
    if not isinstance(result, dict):
        raise C4Unavailable("C4 worker returned an unexpected response.")
    return result


def catalog():
    return _request("/api/c4/info")


def recorded_results():
    return _request("/api/c4/research-results")


def evaluate(case_id, method="lightgbm", family="adaptive"):
    return _request("/api/c4/release/evaluate", {
        "case_id": case_id, "method": method, "family": family,
    })


def fixtures():
    return _request("/api/c4/acceptance/verify", {})
