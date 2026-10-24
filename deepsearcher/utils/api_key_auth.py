"""Opt-in API key authentication for the FastAPI server in main.py."""

import secrets

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests that do not carry the configured ``X-API-Key`` header.

    main.py installs this middleware only when ``--api-key`` is passed, so the
    server keeps its previous unauthenticated behavior by default. Because the
    check runs per request, it also covers routes that were registered before
    the middleware was added (main.py registers its routes at import time) and
    the generated ``/docs`` / ``/openapi.json`` routes.
    """

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        provided = request.headers.get("x-api-key")
        if provided is None or not secrets.compare_digest(
            provided.encode(), self._api_key.encode()
        ):
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid API key."})
        return await call_next(request)
