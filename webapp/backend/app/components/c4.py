"""
C4 Release Controller (owner: Kesigan M, IT23210974). Pipeline step = HTTP call to the C4 worker.

C4 runs in its own Python environment (the "C4 worker", webapp/backend/c4_worker.py, port 8004), so the
main backend never imports C4's libraries. This step sends the pipeline's upstream outputs to

    POST {C4_WORKER_URL}/api/c4/pipeline
    body:     {"context": {...}, "c1": {...}, "c2_to_c4": {...}, "c3": {...}}
    response: {"c4": {...}}            C4 contract v1.1 (shared/contracts/c4_output_schema.json)

If the worker is not running, has no /api/c4/pipeline yet, or returns an error, run() raises and the
orchestrator uses the contract example instead: the step is shown "fake" with the reason and the
pipeline still completes. A decision is never made up here.
"""
import json
import math
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from ..settings import C4_TIMEOUT_SEC, C4_WORKER_URL
from .base import Component

SENT = ("context", "c1", "c2_to_c4", "c3")      # what C4 consumes (C4 contract inputs)


class C4WorkerError(RuntimeError):
    pass


def _clean(value):
    """JSON-safe copy: NaN/inf -> None (C4 rejects non-finite numbers)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):   # numpy scalars
        return _clean(value.item())
    return value


def _local_url(base):
    u = urlsplit(base)
    if u.scheme != "http" or u.hostname not in ("127.0.0.1", "localhost") or u.path not in ("", "/"):
        raise C4WorkerError("C4_WORKER_URL must be a local http address, e.g. http://127.0.0.1:8004")
    return base


def _call(path, payload=None, timeout=C4_TIMEOUT_SEC):
    url = _local_url(C4_WORKER_URL) + path
    data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
    req = Request(url, data=data, method="GET" if payload is None else "POST",
                  headers={"Accept": "application/json", "Content-Type": "application/json"})
    opener = build_opener(ProxyHandler({}))       # local call: never through a system proxy
    try:
        with opener.open(req, timeout=timeout) as r:
            body = r.read()
    except HTTPError as e:
        detail = ""
        try:
            detail = json.loads(e.read() or b"{}").get("detail", "")
        except Exception:  # noqa: BLE001
            pass
        if e.code == 404:
            raise C4WorkerError("the C4 worker has no /api/c4/pipeline endpoint yet") from e
        raise C4WorkerError(f"C4 worker returned HTTP {e.code}{': ' + str(detail) if detail else ''}") from e
    except (URLError, OSError) as e:
        raise C4WorkerError(f"cannot reach the C4 worker at {C4_WORKER_URL} (is its terminal running?)") from e
    try:
        out = json.loads(body)
    except ValueError as e:
        raise C4WorkerError("the C4 worker returned invalid JSON") from e
    if not isinstance(out, dict):
        raise C4WorkerError("the C4 worker returned an unexpected response")
    return out


class C4Component(Component):
    name = "c4"
    title = "C4 Release Controller"
    owner = "Kesigan M"
    produces = ("c4",)
    source = "real"

    def load(self):
        pass                                      # nothing to load here; the worker loads C4's models

    def details(self):
        try:
            h = _call("/health", timeout=2)
            return {"worker_url": C4_WORKER_URL, "worker": "reachable", "worker_health": h}
        except C4WorkerError as e:
            return {"worker_url": C4_WORKER_URL, "worker": f"not reachable: {e}"}

    def status(self):
        # The sidebar shows "real" only while the worker answers; otherwise every run falls back.
        s = super().status()
        if s.get("worker") != "reachable":
            s["source"] = "fake"
        return s

    def run(self, batch, upstream):
        payload = _clean({k: upstream.get(k) for k in SENT})
        out = _call("/api/c4/pipeline", payload)
        if "c4" not in out or not isinstance(out["c4"], dict):
            raise C4WorkerError("the C4 worker response has no 'c4' object")
        return {"c4": out["c4"]}
