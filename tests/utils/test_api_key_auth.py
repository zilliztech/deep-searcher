"""Tests for the opt-in API key middleware used by main.py's --api-key flag."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from deepsearcher.utils.api_key_auth import ApiKeyMiddleware


def _build_client(api_key):
    """Build a client mirroring main.py's wiring.

    main.py registers every route at import time and only adds the middleware later,
    in the ``__main__`` block once argparse has run, so the routes here are declared
    before the middleware is installed as well.
    """
    app = FastAPI()

    @app.get("/probe")
    def probe():
        return {"ok": True}

    if api_key is not None:
        app.add_middleware(ApiKeyMiddleware, api_key=api_key)

    return TestClient(app)


def test_no_api_key_keeps_server_open():
    assert _build_client(None).get("/probe").status_code == 200


def test_missing_header_is_rejected():
    assert _build_client("secret").get("/probe").status_code == 401


def test_wrong_key_is_rejected():
    client = _build_client("secret")
    assert client.get("/probe", headers={"X-API-Key": "wrong"}).status_code == 401


def test_correct_key_is_accepted():
    response = _build_client("secret").get("/probe", headers={"X-API-Key": "secret"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_non_ascii_header_is_rejected_without_server_error():
    # secrets.compare_digest raises TypeError on non-ASCII str input, so the header
    # value must be compared as bytes; an invalid key stays a 401, not a 500
    response = _build_client("secret").get(
        "/probe", headers={b"X-API-Key": "é".encode("latin-1")}
    )
    assert response.status_code == 401


def test_docs_and_openapi_are_covered():
    client = _build_client("secret")
    assert client.get("/docs").status_code == 401
    assert client.get("/openapi.json").status_code == 401


def test_cors_preflight_is_not_blocked_by_auth():
    app = FastAPI()

    @app.get("/probe")
    def probe():
        return {"ok": True}

    # same order as main.py: auth middleware first, CORS added afterwards, which
    # makes CORS the outermost middleware and lets it answer preflight requests
    app.add_middleware(ApiKeyMiddleware, api_key="secret")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    response = TestClient(app).options(
        "/probe",
        headers={"Origin": "http://example.com", "Access-Control-Request-Method": "GET"},
    )
    assert response.status_code == 200
