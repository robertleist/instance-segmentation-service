from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.dependencies import get_current_backend

PUBLIC_PATHS = {
    "/docs",
    "/openapi.json",
    "/redoc",
    "/health",
    "/register",
}


class BackendAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        try:
            get_current_backend(request.headers.get("Authorization"))
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

        return await call_next(request)
