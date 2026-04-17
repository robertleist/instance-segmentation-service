from contextlib import asynccontextmanager
from logging import DEBUG
from logging import getLogger

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from paths import SERVICE_NAME, SERVICE_DESCRIPTION, ALLOWED_ORIGINS

# Router imports
from app.routes import router as health_router
from app.routes.inference import router as inference_router
from app.routes.training import router as training_router

logger = getLogger(__name__)
logger.setLevel(DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.debug("Starting up the Prompted Segmentation Service")
    logger.debug("Celery initialized")
    yield
    # Shutdown code
    logger.debug("Shutting down the Prompted Segmentation Service")


def create_app():
    logger.debug("Creating FastAPI application")
    # Load environment variables
    load_dotenv()

    app = FastAPI(
        title=SERVICE_NAME,
        lifespan=lifespan,
        description=SERVICE_DESCRIPTION,
        version="0.1.0",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include the routers
    app.include_router(health_router)
    app.include_router(inference_router)
    app.include_router(training_router)

    return app