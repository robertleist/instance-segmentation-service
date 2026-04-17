import secrets

from fastapi import Header, HTTPException

from app.state import Backend, get_backend, get_backend_token


def get_current_backend(authorization: str | None = Header(default=None)) -> Backend:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    expected_token = get_backend_token()
    backend = get_backend()
    if (
        backend is None
        or expected_token is None
        or not secrets.compare_digest(token, expected_token)
    ):
        raise HTTPException(status_code=401, detail="Invalid backend token")
    return backend
