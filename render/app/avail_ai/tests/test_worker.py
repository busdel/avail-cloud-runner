"""Worker job-server tests: auth, queueing, sandboxing basics."""
import io
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avail_omics.worker import JobStore, make_handler, new_token  # noqa: E402


def _tiny_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("t/metadata.tsv", "SampleID\tGroup\ns1\tA\ns2\tB\n")
        zf.writestr("t/features/m1.tsv",
                    "SampleID\tf1\ns1\t1.0\ns2\t2.0\n")
    return buf.getvalue()


@pytest.fixture()
def server(tmp_path):
    token = new_token()
    store = JobStore(tmp_path / "work", max_parallel=1)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0),
                                make_handler(store, token, "test"))
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{port}", token, store
    httpd.shutdown()


def _req(method, url, token=None, data=None):
    r = urllib.request.Request(url, method=method)
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        r.data = data
    return urllib.request.urlopen(r, timeout=10)


def test_health_is_public(server):
    base, token, store = server
    with _req("GET", base + "/health") as r:
        assert r.status == 200
        assert b"test" in r.read()


def test_jobs_require_token(server):
    base, token, store = server
    with pytest.raises(urllib.error.HTTPError) as e1:
        _req("GET", base + "/jobs/x")
    assert e1.value.code == 401
    with pytest.raises(urllib.error.HTTPError) as e2:
        _req("GET", base + "/jobs/x", token="wrong")
    assert e2.value.code == 403


def test_submit_and_lifecycle(server):
    base, token, store = server
    with _req("POST", base + "/jobs?flags=robustness", token,
              _tiny_zip()) as r:
        assert r.status == 202
        import json
        jid = json.loads(r.read())["id"]
    with _req("GET", f"{base}/jobs/{jid}", token) as r:
        j = json.loads(r.read())
    assert j["status"] in ("queued", "running", "error")
    assert j["flags"] == ["robustness"]
    with _req("GET", f"{base}/jobs/{jid}/log", token) as r:
        assert r.status == 200
    with pytest.raises(urllib.error.HTTPError) as e:
        _req("GET", f"{base}/jobs/{jid}/results.zip", token)
    assert e.value.code in (404,)
    with _req("DELETE", f"{base}/jobs/{jid}", token) as r:
        assert b"true" in r.read()


def test_reject_non_zip_body(server):
    base, token, store = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _req("POST", base + "/jobs", token, b"not a zip")
    assert e.value.code == 400
