# Template Repository
This is a template repository meant to be used to add new services to the IQUANA annotation tool. You can copy this repository to implement your own service for the tool. 
>[!IMPORTANT]
> With this repo you can only implement your own service to run on its own. To be able to use it in the annotation tool, you still need to add it to the main API and depending on your service the frontend (to add the needed inputs). 
## Installation & Packages
This repository is built with [uv](https://docs.astral.sh/uv/). I recommend using uv to install and manage dependencies. Just run the following and you are good to go:
```shell
uv sync
```
Adding a new package is as easy as running:
```shell
uv add numpy
```
or
````shell
uv pip install numpy
````
> [!TIP]
> If you need more complex setup options like installing from a repo, consider dockerizing.

For AI training features, ensure Redis is running (default broker). Start Celery workers with: `celery -A celery_app worker --loglevel=info`. Monitor tasks with Flower: `celery -A celery_app flower`.

MLFlow is used for model registry. By default, it uses a local SQLite database. For production, configure a remote tracking server.

## Service Registration
This service is designed to be registered with a single main backend at runtime. The backend sends configuration via a POST request to `/register` with the following:

```json
{
  "backend_url": "http://main-backend:8000",
  "celery_broker_url": "redis://celery-host:6379/0",
  "mlflow_tracking_uri": "http://mlflow-host:5000",
  "registration_token": "your-secret-token",
   "service_name": "segmentation-service"
}
```

### Setting up Registration

1. **Before starting the service**, set the registration token (must match the token sent by the backend):
   ```bash
   export SERVICE_REGISTRATION_TOKEN="your-secret-token"
   ```
   If not set, defaults to `SERVICE_SECRET` env var, or `"default-secret"` as fallback.

2. **Start the service**:
   ```bash
   uv run fastapi run main.py
   ```

3. **From the main backend**, call the registration endpoint:
   ```python
   import requests
   
   response = requests.post(
       "http://service-ip:8000/register",
       json={
           "backend_url": "http://main-backend:8000",
           "celery_broker_url": "redis://celery-host:6379/0",
           "mlflow_tracking_uri": "http://mlflow-host:5000",
           "registration_token": "your-secret-token"
       }
   )
   print(response.json())
   ```

4. **Use the returned bearer token** for all subsequent requests:
   ```bash
   curl -H "Authorization: Bearer <backend-token>" http://service:8000/inference
   ```

See [examples/registration_client.py](examples/registration_client.py) for a complete example.

### How Registration Works

1. Backend sends registration request with Celery broker and MLFlow tracking URIs
2. Service validates the registration token
3. Service dynamically configures Celery to use the provided broker
4. Service stores the backend configuration in memory and issues a bearer token
5. Subsequent requests authenticate with `Authorization: Bearer <token>`
6. Training tasks use the registered MLFlow tracking URI when logging runs

## Authentication

Two tokens are involved in the current flow:

### 1. Registration Token
- **When**: During service registration (`POST /register`)
- **How**: 
  - Set on service startup via `SERVICE_REGISTRATION_TOKEN` env var
  - Backend includes token in the registration request
  - Service validates it before accepting the backend configuration
  - Defined in `app/state.py`

### 2. Backend Bearer Token
- **When**: After registration, on all protected requests
- **How**:
  - Issued by the service in the `/register` response
  - Stored by the backend and sent in the `Authorization` header
  - Example:
    ```bash
    curl -H "Authorization: Bearer <backend-token>" http://service:8000/inference
    ```

### Authentication Flow Diagram
```
1. Service starts (waiting for registration)
   ↓
2. Backend sends registration request with token
   ↓
3. Service validates registration token → Stores backend configuration
   ↓
4. Service returns a backend bearer token
   ↓
5. Requests without a valid bearer token are rejected
```

### Security Recommendations
- Use strong registration tokens (e.g., long random strings)
- Set `SERVICE_REGISTRATION_TOKEN` at service startup, not in code
- Treat the returned backend bearer token like a secret
- Use HTTPS for all service communication in production
- Keep MLFlow and Redis ports behind firewalls
- Consider network segmentation for service-to-backend communication
## Structure
This template services uses [FastAPI](https://fastapi.tiangolo.com) for routing and managing HTTP endpoints. FastAPI is a modern, fast (high-performance), web framework for building APIs with Python based on standard Python type hints.
You can see the docs, when you run the API:
```shell
uv run fastapi run main.py
```

Each service is its own API in the coral setup, accepting (and sending) requests. In general, a lot of services implement the same functionality, hence this repo is meant to make creating a new service easier by supplying a skeleton with most functionalities implemented.

### Where to find what
All **endpoints** can be found under [app/routes](app/routes). An endpoint to check whether the service is running, is defined in [app/routes/\_\_init__.py](app/routes/__init__.py).

[Pydantic](https://docs.pydantic.dev/latest/) schemas should be added under [app/schemas](app/schemas). These help you to define data classes easily. You can directly pass them to fastapi routes and they are json serializable. 
Alternatively, add the schemas to your instance of the [IQUANA toolbox](https://github.com/Iquana-tool/iquana-toolbox). This way you can make use of them across services and the main backend.

**Caches and persistent memory** should be added to [app/state.py](app/state.py). 

The **app creator** function can be found under [app/\_\_init\_\_.py](app/\_\_init\_\_.py). You need to add new routers here, if you create any.

**Utility functionality** should be added under [util](util). It already contains an image cache and loading functionality.

Any functionality regarding **models** should be added under [models](models). This folder already contains a bunch of .py files. Notably, a base model class, a registry, a cache and a register script.

## Workflow and Endpoints
### Selecting a model
Before you run inference you need to select a model, which you want to use. This is intentionally separated from the actual inference call, as users might choose the model way before actually choosing to send a request. Giving the backend time to preload the model.
- `list_models`: Get a list of all available (and loadable) models. This is important for the main API to give the user a list of possible models.
- `load_model`: Load a model into the model cache, such that the user can later on use it.
#### Model Registry
The service uses MLFlow for model management, including registration, versioning, and staging. Models are tracked with experiments, runs, and metadata. The MLFlow tracking URI is provided by the backend during registration and then reused for later requests and training tasks. You can register models by editing the [registration function](models/register_models.py). 

Each model has metadata like name, description, tags, and training config. MLFlow handles loading and caching automatically.
### Inference
If image and model are selected and loaded, you can run inference. This endpoint does not come with predefined functionality. You need to implement it yourself. The route already resolves the registered backend through a FastAPI dependency so you can access `backend.mlflow_tracking_uri` and other backend metadata directly.
### Training
For AI training, the service now supports asynchronous model training using Celery for background task processing.
1. Upload a dataset (similar to images).
2. Select a model with training support.
3. Start training with parameters (e.g., epochs, learning rate).
4. Monitor progress and retrieve results.

The training route currently accepts a raw JSON body as a placeholder. Replace it with a service-specific Pydantic schema once you define your training API.

Endpoints:
- `POST /train`: Start a training job (returns task ID).
- `DELETE /train/{task_id}`: Cancel a training job.

Training uses Celery workers for scalability. The service configures the global Celery app with the registered backend's broker during `/register`. Run workers with: `celery -A celery_app worker --loglevel=info`. Monitor with Flower: `celery -A celery_app flower`.
