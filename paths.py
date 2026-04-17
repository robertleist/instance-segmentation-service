from os import getenv
import os
"""
In this file you can add paths you regularly access or load environment variables.
"""

# General paths
ROOT = os.path.dirname(os.path.realpath(__file__))  # Your project root
EXAMPLES_DIR = os.path.join(ROOT, "examples")  # A dir called "examples" under your root dir
LOG_DIR = getenv("LOG_DIR", "logs")  # Loads the dir from your .env file

# Configs
SERVICE_NAME = getenv("SERVICE_NAME", "Template annotation service")
SERVICE_DESCRIPTION = getenv("SERVICE_DESCRIPTION", "A template service for the annotation tool.")
MLFLOW_URL = getenv("ML_FLOW_URL", "http://localhost:5000")
REDIS_URL = getenv("REDIS_URL", "redis://localhost:6379")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
