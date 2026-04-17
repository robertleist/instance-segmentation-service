from celery import Celery
from paths import SERVICE_NAME, REDIS_URL

# Configure Celery app
app = Celery(
    SERVICE_NAME,
    broker=REDIS_URL,  # Or use env var for flexibility
    backend=REDIS_URL,
    include=["app.tasks"]  # Where training tasks will be defined
)

# Optional: Auto-discover tasks
app.autodiscover_tasks()