# AGENTS.md

## Overview
- This repo is a FastAPI/Celery template for a **single IQUANA service** that **registers itself with exactly one backend at runtime**; the central flow is `/register` → in-memory backend state → protected `/inference` and `/train` calls.
- The entry point is `main.py`; `create_app()` in `app/__init__.py` wires up CORS, authentication middleware, and routers.
- The actual backend configuration does not live in FastAPI state; it is stored as **module-level globals** in `app/state.py` (`_backend`, `_backend_token`). Re-registration replaces the previous configuration.

## Key runtime flows
- `POST /register` in `app/routes/registration.py` expects `iquana_toolbox.schemas.networking.http.services.ServiceRegistrationRequest`.
- Registration validates `SERVICE_REGISTRATION_TOKEN`, then reconfigures the global Celery broker through `app.state.update_celery_config()`, and finally issues a new bearer token via `register_backend()`.
- Protected endpoints use **Authorization: Bearer <token>**. This is enforced both in `app/middleware.py` and again in `app/dependencies.py::get_current_backend()`.
- `get_current_backend()` returns a `Backend` dataclass object; new endpoints should reuse this dependency instead of parsing headers manually.
- `POST /train` forwards `model_id`, `dataset_path`, `params`, and `backend.mlflow_tracking_uri` to `app.tasks.train_model.delay(...)`.

## Actual endpoints in the current codebase
- `GET /health` in `app/routes/__init__.py`
- `POST /register` in `app/routes/registration.py`
- `POST /inference` in `app/routes/inference.py` (still raises `NotImplementedError`)
- `POST /train` and `DELETE /train/{task_id}` in `app/routes/training.py`

## Project conventions and gotchas
- This template is **partially inconsistent**: `app/__init__.py` imports `MODEL_REGISTRY` from `app/state.py` and `models.register_models`, but both are currently missing. A simple import of `main.py` fails because of that.
- When changing behavior, trust the **code** more than generic README statements: `README.md` mentions model-registry files, `app/schemas/`, and utility code that do not currently exist in this workspace.
- `examples/registration_client.py` is outdated: it sends `X-API-Key`, while the middleware only accepts `Authorization: Bearer ...`.
- Middleware bypass is path-based (`/docs`, `/openapi.json`, `/item`, `/health`, `/register`); if you add new public endpoints, you must update `app/middleware.py`.
- Logging is initialized in `main.py` and written to `logs/logs.txt`; check that file and the Uvicorn/FastAPI console first when debugging.

## External integrations
- Python/dependency management uses `uv`; dependencies are declared in `pyproject.toml`.
- `iquana-toolbox` is pulled directly from Git and provides the HTTP schemas used for registration and image requests.
- Celery is defined globally in `celery_app.py`, starts with local Redis defaults, and is only reconfigured to the backend broker after `/register`.
- MLflow is used directly inside training tasks; `train_model()` sets the tracking URI per task and logs artifacts/models there.

## Verified workflows
- Installation according to `README.md`:
  - `uv sync`
- Start the API according to `README.md`:
  - `uv run fastapi run main.py`
- Worker/monitoring according to `README.md`:
  - `celery -A celery_app worker --loglevel=info`
  - `celery -A celery_app flower`
- There are currently **no tests** under `tests/` or as `test_*.py`; validate changes at minimum with targeted imports and smoke checks.

## If you add new features
- Attach new routers in `app/__init__.py` inside `create_app()`.
- If an endpoint is backend-bound, use `backend: Backend = Depends(get_current_backend)` as shown in `app/routes/inference.py` and `app/routes/training.py`.
- For service-specific training APIs, replace the raw `dict` body in `app/routes/training.py` with a schema before expanding task logic.
- Fix the missing template pieces (`MODEL_REGISTRY`, `models/register_models.py`) before extending model or inference workflows.
