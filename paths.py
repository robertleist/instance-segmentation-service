from os import getenv
import os
from dotenv import load_dotenv
"""
In this file you can add paths you regularly access or load environment variables.
"""

# General paths
ROOT = os.path.dirname(os.path.realpath(__file__))  # Your project root

# Load variables from the project's .env into the environment so getenv() below
# (and anything imported later) can see them.
load_dotenv(os.path.join(ROOT, ".env"))
EXAMPLES_DIR = os.path.join(ROOT, "examples")  # A dir called "examples" under your root dir
LOG_DIR = getenv("LOG_DIR", "logs")  # Loads the dir from your .env file

# Configs
SERVICE_NAME = getenv("SERVICE_NAME", "Template annotation service")
SERVICE_DESCRIPTION = getenv("SERVICE_DESCRIPTION", "A template service for the annotation tool.")
# Empty/blank -> None so transformers does an anonymous request instead of
# sending an illegal "Authorization: Bearer " header (httpcore LocalProtocolError).
HF_ACCESS_TOKEN = getenv("HF_ACCESS_TOKEN", None)
MLFLOW_URL = getenv("ML_FLOW_URL", "http://localhost:5000")
REDIS_URL = getenv("REDIS_URL", "redis://localhost:6379")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
